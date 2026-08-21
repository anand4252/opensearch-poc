"""API routes for the neural-sparse POC (BM25 + sparse, doc-only, hybrid)."""

from fastapi import APIRouter, HTTPException

from app.deps import client, load_seed_docs, settings
from app.ingest import bulk_index, index_count
from app.models import (
    BootstrapResponse,
    IngestRequest,
    IngestResponse,
    SearchMode,
    SearchRequest,
    SearchResponse,
)
from app.sparse.bootstrap import run_bootstrap
from app.sparse.search import run_search

router = APIRouter(prefix="/sparse", tags=["sparse"])


@router.post("/bootstrap", response_model=BootstrapResponse)
def bootstrap(recreate_index: bool = False, persist: bool = True) -> BootstrapResponse:
    """Bootstrap the sparse cluster: ML settings, sparse model, ingest/search pipelines, index.

    Same logic as `make bootstrap`; exposed here so you can step through it in a
    debugger. `recreate_index` drops & rebuilds the index if present. `persist` writes
    the resolved MODEL_ID to .env (pass false to leave .env untouched while debugging).
    """
    try:
        result = run_bootstrap(
            client(), settings(), recreate_index=recreate_index, persist=persist
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return BootstrapResponse(**result)


@router.post("/documents", response_model=IngestResponse)
def add_documents(req: IngestRequest) -> IngestResponse:
    """Ad-hoc ingest of arbitrary documents into the sparse index."""
    docs = [d.model_dump() for d in req.documents]
    success, errors = bulk_index(client(), settings(), docs)
    return IngestResponse(
        indexed=success, errors=errors, total_in_index=index_count(client(), settings())
    )


@router.post("/seed", response_model=IngestResponse)
def seed_documents() -> IngestResponse:
    """Index the prepared dataset (data/flickr_docs.json) into the sparse index."""
    docs = load_seed_docs()
    success, errors = bulk_index(client(), settings(), docs)
    return IngestResponse(
        indexed=success, errors=errors, total_in_index=index_count(client(), settings())
    )


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    """Lexical / sparse / hybrid search over the sparse index."""
    try:
        return run_search(client(), settings(), req.query, req.mode, req.size)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/search", response_model=SearchResponse)
def search_get(q: str, mode: SearchMode = SearchMode.hybrid, size: int = 5) -> SearchResponse:
    """Convenience GET variant, e.g. /sparse/search?q=a%20dog%20on%20the%20beach&mode=hybrid"""
    return search(SearchRequest(query=q, mode=mode, size=size))
