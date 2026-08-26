"""Multimodal search: kNN over CLIP image vectors, queried by text OR by an image.

Because CLIP puts images and text in one shared space, the *same* index answers both:
- text -> image: embed the query text, find nearest image vectors.
- image -> image: embed a query image, find nearest image vectors (visually similar).

Either way we embed the query in our app and pass the vector to a plain `knn` query.
"""

from __future__ import annotations

from app.config import Settings
from app.models import MultimodalSearchResponse, SearchHit
from app.multimodal.clip_model import embed_image_bytes, embed_text


def _knn_search(client, settings: Settings, vector: list[float], size: int) -> list[SearchHit]:
    body = {
        "size": size,
        "_source": {"excludes": [settings.image_vector_field]},
        "query": {"knn": {settings.image_vector_field: {"vector": vector, "k": size}}},
    }
    resp = client.search(index=settings.multimodal_index_name, body=body)
    return [
        SearchHit(
            score=h["_score"],
            name=h["_source"].get("name", ""),
            combined_text=h["_source"].get(settings.text_field, ""),
        )
        for h in resp["hits"]["hits"]
    ]


def run_text_search(
    client, settings: Settings, query: str, size: int
) -> MultimodalSearchResponse:
    """Text -> image: embed the query text with CLIP, find the nearest images."""
    vector = embed_text(settings, query)
    return MultimodalSearchResponse(query=query, hits=_knn_search(client, settings, vector, size))


def run_image_search(
    client, settings: Settings, image_bytes: bytes, size: int, *, label: str
) -> MultimodalSearchResponse:
    """Image -> image: embed a query image with CLIP, find visually similar images."""
    vector = embed_image_bytes(settings, image_bytes)
    return MultimodalSearchResponse(query=label, hits=_knn_search(client, settings, vector, size))
