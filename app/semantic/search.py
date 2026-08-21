"""Dense semantic search: the `neural` query (kNN over dense embeddings).

The `neural` query encodes `query` with the dense model at SEARCH time, then does an
ANN nearest-neighbor lookup against the `knn_vector` field. This is the defining
contrast with sparse doc-only mode, which does no query-time model inference.
"""

from __future__ import annotations

from app.config import Settings
from app.ml_commons import find_deployed_model
from app.models import DenseSearchResponse, SearchHit


def _resolve_model_id(client, settings: Settings) -> str:
    """Prefer the configured DENSE_MODEL_ID; fall back to the live deployed model.

    The fallback means search keeps working right after a bootstrap even though the
    running app's cached settings haven't picked up the newly persisted .env value.
    """
    if settings.dense_model_id:
        return settings.dense_model_id
    model_id = find_deployed_model(client, settings.dense_model_name)
    if not model_id:
        raise RuntimeError(
            "No deployed dense model found. Run the dense bootstrap first: "
            "`make bootstrap-semantic` or POST /semantic/bootstrap."
        )
    return model_id


def run_dense_search(client, settings: Settings, query: str, size: int) -> DenseSearchResponse:
    model_id = _resolve_model_id(client, settings)
    body = {
        "_source": {"excludes": [settings.knn_field]},
        "size": size,
        "query": {
            "neural": {
                settings.knn_field: {
                    "query_text": query,
                    "model_id": model_id,
                    "k": size,
                }
            }
        },
    }
    resp = client.search(index=settings.dense_index_name, body=body)
    hits = [
        SearchHit(
            score=h["_score"],
            name=h["_source"].get("name", ""),
            combined_text=h["_source"].get(settings.text_field, ""),
        )
        for h in resp["hits"]["hits"]
    ]
    return DenseSearchResponse(query=query, model_id=model_id, hits=hits)
