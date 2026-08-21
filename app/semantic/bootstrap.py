"""Bootstrap the dense semantic search POC.

Parallel to the neural-sparse bootstrap (`app.sparse.bootstrap`) but for DENSE vectors:

  - Ingest time: a text-embedding model (msmarco-distilbert-base-tas-b) encodes each
    document's text into one 768-dim dense vector, stored in a `knn_vector` field.
  - Query time : the SAME model encodes the query text (the `neural` query runs model
    inference at search time), then OpenSearch does an ANN (HNSW) nearest-neighbor
    search. This query-time inference is the key contrast with sparse doc-only mode.

Generic ML Commons plumbing (register/deploy/reuse, .env persistence) is shared from
`app.ml_commons`; this module holds only the dense-specific pipeline/index wiring.

Steps (idempotent — safe to re-run):
  1. Apply ML Commons cluster settings.
  2. Register + deploy the dense embedding model, reusing it if already deployed.
  3. Create the ingest pipeline (text_embedding processor).
  4. Create the knn index (knn_vector field, default_pipeline).
  5. Optionally persist the resolved model_id into .env (DENSE_MODEL_ID=...).
"""

from __future__ import annotations

from app.ml_commons import apply_ml_settings, persist_env_var, register_and_deploy


def register_and_deploy_model(client, settings) -> tuple[str, bool]:
    """Register + deploy the dense text-embedding model, reusing it if already deployed."""
    return register_and_deploy(
        client,
        name=settings.dense_model_name,
        version=settings.dense_model_version,
        model_format=settings.dense_model_format,
    )


def create_ingest_pipeline(client, settings, model_id: str) -> None:
    print(f"\n[3/5] Creating dense ingest pipeline '{settings.dense_ingest_pipeline}'...")
    client.ingest.put_pipeline(
        id=settings.dense_ingest_pipeline,
        body={
            "description": "Text-embedding pipeline for dense semantic search",
            "processors": [
                {
                    "text_embedding": {
                        "model_id": model_id,
                        "field_map": {settings.text_field: settings.knn_field},
                    }
                }
            ],
        },
    )
    print("      done.")


def create_index(client, settings, recreate: bool) -> bool:
    """Create the knn index. Returns True if it was (re)created, False if left as-is."""
    print(f"\n[4/5] Creating dense index '{settings.dense_index_name}'...")
    if client.indices.exists(index=settings.dense_index_name):
        if recreate:
            print("      exists -> deleting (recreate_index).")
            client.indices.delete(index=settings.dense_index_name)
        else:
            print("      already exists -> skipping (use recreate_index to rebuild).")
            return False

    client.indices.create(
        index=settings.dense_index_name,
        body={
            "settings": {
                "index.knn": True,
                "default_pipeline": settings.dense_ingest_pipeline,
                # Single node: 1 shard / 0 replicas keeps cluster health green.
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "mappings": {
                "properties": {
                    "name": {"type": "text"},
                    settings.text_field: {"type": "text"},  # keep the raw caption
                    settings.knn_field: {
                        "type": "knn_vector",
                        "dimension": settings.embedding_dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": settings.dense_space_type,
                            "engine": settings.dense_engine,
                        },
                    },
                }
            },
        },
    )
    print("      done.")
    return True


def run_bootstrap(client, settings, *, recreate_index: bool = False, persist: bool = True) -> dict:
    """Run the dense bootstrap steps and return a structured summary.

    `recreate_index` drops & rebuilds the dense index if it exists. `persist` writes the
    resolved DENSE_MODEL_ID back to .env (skip it when debugging).
    """
    apply_ml_settings(client)
    model_id, model_reused = register_and_deploy_model(client, settings)
    create_ingest_pipeline(client, settings, model_id)
    index_created = create_index(client, settings, recreate=recreate_index)
    if persist:
        persist_env_var("DENSE_MODEL_ID", model_id)

    return {
        "model_id": model_id,
        "model_reused": model_reused,
        "index_created": index_created,
        "ingest_pipeline": settings.dense_ingest_pipeline,
        "search_pipeline": None,  # pure dense: the `neural` query needs no search pipeline
        "persisted": persist,
    }
