"""Bootstrap a local OpenSearch cluster for hybrid (BM25 + neural sparse) search.

Reproduces, locally, the AWS setup this project is modelled on. Neural SPARSE search
in *doc-only* mode:

  - Ingest time: a sparse-encoding model expands each document's text into a
    map of token -> weight, stored in a `rank_features` field.
  - Query time : a built-in analyzer ("bert-uncased") tokenizes the query, so there
    is NO model inference at search time (low latency). Supported on OpenSearch 3.x.

Steps (idempotent — safe to re-run):
  1. Apply ML Commons cluster settings.
  2. Register + deploy the sparse ingest model, capture its model_id.
  3. Create the ingest pipeline (sparse_encoding processor).
  4. Create the index (text + rank_features fields, default_pipeline).
  5. Create the search pipeline (normalization-processor for hybrid scoring).
  6. Persist the resolved model_id into .env (MODEL_ID=...).

Run:  uv run python scripts/bootstrap_opensearch.py
      uv run python scripts/bootstrap_opensearch.py --recreate-index   # drop & recreate index
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as a plain script: `python scripts/bootstrap_opensearch.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.opensearch_client import build_client  # noqa: E402

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


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


def _find_deployed_model(client, model_name: str) -> str | None:
    """Return the id of an already-deployed model with this name, if any."""
    try:
        res = client.transport.perform_request(
            "POST",
            "/_plugins/_ml/models/_search",
            body={
                "query": {"term": {"name.keyword": model_name}},
                "size": 5,
            },
        )
    except Exception:
        return None
    for hit in res.get("hits", {}).get("hits", []):
        src = hit.get("_source", {})
        if src.get("model_state") == "DEPLOYED":
            return hit["_id"]
    return None


def register_and_deploy_model(client, settings) -> str:
    """Register + deploy the sparse ingest model, reusing it if already deployed.

    Doc-only mode needs only this one model: it expands documents into sparse
    token->weight features at ingest time. The query side uses a built-in analyzer
    (OpenSearch 3.x), so no query-side model is required.
    """
    print("\n[2/6] Registering + deploying sparse ingest model...")
    print(f"      model : {settings.model_name}")
    existing = _find_deployed_model(client, settings.model_name)
    if existing:
        print(f"      reusing already-deployed model: {existing}")
        return existing

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
    return model_id


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


def create_index(client, settings, recreate: bool) -> None:
    print(f"\n[4/6] Creating index '{settings.index_name}'...")
    if client.indices.exists(index=settings.index_name):
        if recreate:
            print("      exists -> deleting (--recreate-index).")
            client.indices.delete(index=settings.index_name)
        else:
            print("      already exists -> skipping (use --recreate-index to rebuild).")
            return

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap OpenSearch hybrid sparse search")
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="Drop and recreate the index if it already exists.",
    )
    args = parser.parse_args()

    settings = get_settings()
    client = build_client(settings)

    try:
        health = client.cluster.health()
    except Exception as exc:  # pragma: no cover - connection error path
        print(f"ERROR: cannot reach OpenSearch at {settings.opensearch_url}: {exc}")
        print("Is the stack up?  make up   (then wait for it to become healthy)")
        return 1
    print(f"Connected. Cluster '{health['cluster_name']}' status: {health['status']}")

    apply_ml_settings(client)
    model_id = register_and_deploy_model(client, settings)
    create_ingest_pipeline(client, settings, model_id)
    create_index(client, settings, recreate=args.recreate_index)
    create_search_pipeline(client, settings)
    persist_model_id(model_id)

    print("\nBootstrap complete.")
    print(f"  model_id        : {model_id}")
    print(f"  index           : {settings.index_name}")
    print(f"  ingest pipeline : {settings.ingest_pipeline}")
    print(f"  search pipeline : {settings.search_pipeline}")
    print("\nNext:  make run   then   make seed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
