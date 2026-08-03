"""Query builders and search execution for lexical / sparse / hybrid modes."""

from app.config import Settings
from app.models import SearchHit, SearchMode, SearchResponse


def _lexical_query(settings: Settings, query: str) -> dict:
    """BM25 keyword match on the text field."""
    return {"match": {settings.text_field: {"query": query}}}


def _sparse_query(settings: Settings, query: str) -> dict:
    """Neural sparse, doc-only mode: the query is tokenized by an analyzer,
    so no model inference happens at search time."""
    return {
        "neural_sparse": {
            settings.embedding_field: {
                "query_text": query,
                "analyzer": settings.query_analyzer,
            }
        }
    }


def build_body(settings: Settings, query: str, mode: SearchMode, size: int) -> dict:
    source = {"excludes": [settings.embedding_field]}
    if mode is SearchMode.lexical:
        return {"_source": source, "size": size, "query": _lexical_query(settings, query)}
    if mode is SearchMode.sparse:
        return {"_source": source, "size": size, "query": _sparse_query(settings, query)}
    # hybrid: two sub-queries combined by the normalization search pipeline
    return {
        "_source": source,
        "size": size,
        "query": {
            "hybrid": {
                "queries": [
                    _lexical_query(settings, query),
                    _sparse_query(settings, query),
                ]
            }
        },
    }


def run_search(
    client, settings: Settings, query: str, mode: SearchMode, size: int
) -> SearchResponse:
    body = build_body(settings, query, mode, size)
    params = {}
    # The normalization pipeline only applies to the hybrid query.
    if mode is SearchMode.hybrid:
        params["search_pipeline"] = settings.search_pipeline

    resp = client.search(index=settings.index_name, body=body, params=params)

    hits = [
        SearchHit(
            score=h["_score"],
            name=h["_source"].get("name", ""),
            combined_text=h["_source"].get(settings.text_field, ""),
        )
        for h in resp["hits"]["hits"]
    ]
    return SearchResponse(mode=mode, query=query, hits=hits)
