"""Multimodal text->image search: embed the query with CLIP, kNN over image vectors.

We embed the query text with the SAME CLIP model used for the images, then ask OpenSearch
for the nearest image vectors with a plain `knn` query (we pass the vector ourselves — no
`neural` query, because OpenSearch isn't running the model here).
"""

from __future__ import annotations

from app.config import Settings
from app.models import MultimodalSearchResponse, SearchHit
from app.multimodal.clip_model import embed_text


def run_multimodal_search(
    client, settings: Settings, query: str, size: int
) -> MultimodalSearchResponse:
    query_vector = embed_text(settings, query)
    body = {
        "size": size,
        "_source": {"excludes": [settings.image_vector_field]},
        "query": {
            "knn": {
                settings.image_vector_field: {"vector": query_vector, "k": size}
            }
        },
    }
    resp = client.search(index=settings.multimodal_index_name, body=body)
    hits = [
        SearchHit(
            score=h["_score"],
            name=h["_source"].get("name", ""),
            combined_text=h["_source"].get(settings.text_field, ""),
        )
        for h in resp["hits"]["hits"]
    ]
    return MultimodalSearchResponse(query=query, hits=hits)
