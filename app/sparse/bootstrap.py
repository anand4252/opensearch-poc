"""Bootstrap logic for the local OpenSearch hybrid (BM25 + neural sparse) setup.

Single source of truth shared by the CLI wrapper (`app.sparse.cli`) and the
`/sparse/bootstrap` endpoint (`app.sparse.routes`). Generic ML Commons plumbing (model
register/deploy, reuse detection, .env persistence) lives in `app.ml_commons`; this
module holds the neural-sparse-specific pipeline/index wiring. Doc-only mode:

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

from app.ml_commons import (
    apply_ml_settings,
    persist_env_var,
    register_and_deploy,
)


def register_and_deploy_model(client, settings) -> tuple[str, bool]:
    """Register + deploy the sparse ingest model, reusing it if already deployed.

    Doc-only mode needs only this one model: it expands documents into sparse
    token->weight features at ingest time. The query side uses a built-in analyzer
    (OpenSearch 3.x), so no query-side model is required.
    """
    return register_and_deploy(
        client,
        name=settings.model_name,
        version=settings.model_version,
        model_format=settings.model_format,
    )


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
        persist_env_var("MODEL_ID", model_id)

    return {
        "model_id": model_id,
        "model_reused": model_reused,
        "index_created": index_created,
        "ingest_pipeline": settings.ingest_pipeline,
        "search_pipeline": settings.search_pipeline,
        "persisted": persist,
    }
