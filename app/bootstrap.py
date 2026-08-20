"""Bootstrap logic for the local OpenSearch hybrid (BM25 + neural sparse) setup.

Single source of truth shared by the CLI wrapper (`scripts/bootstrap_opensearch.py`)
and the `/admin/bootstrap` endpoint (`app.main`). Neural SPARSE search in *doc-only*
mode:

  - Ingest time: a sparse-encoding model expands each document's text into a
    map of token -> weight, stored in a `rank_features` field.
  - Query time : a built-in analyzer ("bert-uncased") tokenizes the query, so there
    is NO model inference at search time (low latency). Supported on OpenSearch 3.x.

Steps (idempotent — safe to re-run):
  1. Apply ML Commons cluster settings.
  2. Register + deploy the sparse ingest model, reusing it if already deployed.
  3. Create the ingest pipeline (sparse_encoding processor).
  4. Create the index (text + rank_features fields, default_pipeline).
  5. Create the search pipeline (normalization-processor for hybrid scoring).
  6. Optionally persist the resolved model_id into .env (MODEL_ID=...).
"""

from __future__ import annotations

import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"


def apply_ml_settings(client) -> None:
    print(
        "\n[1/6] Applying ML Commons cluster settings "
        "(run on single node + raise memory threshold guard)..."
    )
    client.cluster.put_settings(
        body={
            "persistent": {
                "plugins.ml_commons.only_run_on_ml_node": False,
                "plugins.ml_commons.model_access_control_enabled": False,
                "plugins.ml_commons.native_memory_threshold": 99,
            }
        }
    )
    print("      done.")


def _poll_task(client, task_id: str, *, attempts: int = 120, delay: int = 5) -> str:
    """Poll an ML task until COMPLETED; return its model_id."""
    for i in range(attempts):
        task = client.transport.perform_request("GET", f"/_plugins/_ml/tasks/{task_id}")
        state = task.get("state")
        print(f"      [{i * delay:>3}s] task state: {state}")
        if state == "COMPLETED":
            return task["model_id"]
        if state in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"ML task {task_id} {state}: {task.get('error', 'unknown error')}")
        time.sleep(delay)
    raise TimeoutError(f"ML task {task_id} did not complete in time")


def find_deployed_model(client, model_name: str) -> str | None:
    """Return the id of an already-deployed model with this name, if any.

    `.plugins-ml-model` stores each model as one metadata document PLUS N content
    chunks (chunk docs carry `chunk_number`/`total_chunks` but no `model_state`). We
    must exclude the chunks, or every match looks stateless and we'd re-register the
    model on every run. Only the metadata doc has `model_state`; its `_id` is the
    model_id.
    """
    try:
        res = client.transport.perform_request(
            "POST",
            "/_plugins/_ml/models/_search",
            body={
                "query": {
                    "bool": {
                        "must": [{"term": {"name.keyword": model_name}}],
                        "must_not": [{"exists": {"field": "chunk_number"}}],
                    }
                },
                "size": 20,
            },
        )
    except Exception:
        return None
    for hit in res.get("hits", {}).get("hits", []):
        src = hit.get("_source", {})
        if src.get("model_state") == "DEPLOYED":
            return hit["_id"]
    return None


def register_and_deploy_model(client, settings) -> tuple[str, bool]:
    """Register + deploy the sparse ingest model, reusing it if already deployed.

    Returns (model_id, reused). Doc-only mode needs only this one model: it expands
    documents into sparse token->weight features at ingest time. The query side uses a
    built-in analyzer (OpenSearch 3.x), so no query-side model is required.
    """
    print("\n[2/6] Registering + deploying sparse ingest model...")
    print(f"      model : {settings.model_name}")
    existing = find_deployed_model(client, settings.model_name)
    if existing:
        print(f"      reusing already-deployed model: {existing}")
        return existing, True

    resp = client.transport.perform_request(
        "POST",
        "/_plugins/_ml/models/_register?deploy=true",
        body={
            "name": settings.model_name,
            "version": settings.model_version,
            "model_format": settings.model_format,
        },
    )
    task_id = resp["task_id"]
    print(f"      task id: {task_id}")
    model_id = _poll_task(client, task_id)
    print(f"      model deployed: {model_id}")
    return model_id, False


def create_ingest_pipeline(client, settings, model_id: str) -> None:
    print(f"\n[3/6] Creating ingest pipeline '{settings.ingest_pipeline}'...")
    client.ingest.put_pipeline(
        id=settings.ingest_pipeline,
        body={
            "description": "Sparse encoding pipeline (doc-only) for hybrid search",
            "processors": [
                {
                    "sparse_encoding": {
                        "model_id": model_id,
                        "field_map": {settings.text_field: settings.embedding_field},
                    }
                }
            ],
        },
    )
    print("      done.")


def create_index(client, settings, recreate: bool) -> bool:
    """Create the index. Returns True if it was (re)created, False if left as-is."""
    print(f"\n[4/6] Creating index '{settings.index_name}'...")
    if client.indices.exists(index=settings.index_name):
        if recreate:
            print("      exists -> deleting (recreate_index).")
            client.indices.delete(index=settings.index_name)
        else:
            print("      already exists -> skipping (use recreate_index to rebuild).")
            return False

    client.indices.create(
        index=settings.index_name,
        body={
            "settings": {
                "default_pipeline": settings.ingest_pipeline,
                # Single node: 1 shard / 0 replicas keeps cluster health green.
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "mappings": {
                "properties": {
                    "name": {"type": "text"},
                    settings.text_field: {"type": "text"},          # lexical / BM25
                    settings.embedding_field: {"type": "rank_features"},  # neural sparse
                }
            },
        },
    )
    print("      done.")
    return True


def create_search_pipeline(client, settings) -> None:
    print(f"\n[5/6] Creating search pipeline '{settings.search_pipeline}'...")
    client.transport.perform_request(
        "PUT",
        f"/_search/pipeline/{settings.search_pipeline}",
        body={
            "description": "Normalization pipeline for hybrid sparse search",
            "phase_results_processors": [
                {
                    "normalization-processor": {
                        "normalization": {"technique": "min_max"},
                        "combination": {
                            "technique": "arithmetic_mean",
                            "parameters": {
                                # [lexical, sparse]
                                "weights": [settings.lexical_weight, settings.sparse_weight]
                            },
                        },
                    }
                }
            ],
        },
    )
    print("      done.")


def _set_env_var(lines: list[str], key: str, value: str) -> list[str]:
    """Replace `key=...` in .env lines, or append it if not present."""
    prefix = f"{key}="
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            lines[i] = f"{key}={value}"
            return lines
    lines.append(f"{key}={value}")
    return lines


def persist_model_id(model_id: str) -> None:
    print(f"\n[6/6] Persisting MODEL_ID to {ENV_FILE.name}...")
    if not ENV_FILE.exists():
        # Seed from the example if the user hasn't copied it yet.
        example = ENV_FILE.with_name(".env.example")
        ENV_FILE.write_text(example.read_text() if example.exists() else "")

    lines = ENV_FILE.read_text().splitlines()
    lines = _set_env_var(lines, "MODEL_ID", model_id)
    ENV_FILE.write_text("\n".join(lines) + "\n")
    print("      done.")


def run_bootstrap(client, settings, *, recreate_index: bool = False, persist: bool = True) -> dict:
    """Run all bootstrap steps and return a structured summary.

    `recreate_index` drops & rebuilds the index if it exists. `persist` writes the
    resolved MODEL_ID back to .env (skip it when debugging so the file isn't touched).
    """
    apply_ml_settings(client)
    model_id, model_reused = register_and_deploy_model(client, settings)
    create_ingest_pipeline(client, settings, model_id)
    index_created = create_index(client, settings, recreate=recreate_index)
    create_search_pipeline(client, settings)
    if persist:
        persist_model_id(model_id)

    return {
        "model_id": model_id,
        "model_reused": model_reused,
        "index_created": index_created,
        "ingest_pipeline": settings.ingest_pipeline,
        "search_pipeline": settings.search_pipeline,
        "persisted": persist,
    }
