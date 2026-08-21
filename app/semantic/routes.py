"""API routes for the dense semantic POC (text-embedding + knn_vector + `neural`)."""

from fastapi import APIRouter, HTTPException

from app.deps import client, load_seed_docs, settings
from app.ingest import bulk_index, index_count
from app.models import (
    BootstrapResponse,
    DenseSearchRequest,
    DenseSearchResponse,
    IngestResponse,
)
from app.semantic.bootstrap import run_bootstrap as run_dense_bootstrap
from app.semantic.search import run_dense_search

router = APIRouter(prefix="/semantic", tags=["semantic"])


@router.post("/bootstrap", response_model=BootstrapResponse)
def bootstrap(recreate_index: bool = False, persist: bool = True) -> BootstrapResponse:
    """Bootstrap the DENSE semantic index: embedding model, text_embedding pipeline, knn index.

    Parallel to /sparse/bootstrap but for dense vectors. `persist` writes DENSE_MODEL_ID
    to .env. First run downloads/deploys the model (slow); later runs reuse it.
    """
    try:
        result = run_dense_bootstrap(
            client(), settings(), recreate_index=recreate_index, persist=persist
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return BootstrapResponse(**result)


@router.post("/seed", response_model=IngestResponse)
def seed_documents() -> IngestResponse:
    """Index the prepared dataset (data/flickr_docs.json) into the dense index."""
    docs = load_seed_docs()
    index = settings().dense_index_name
    success, errors = bulk_index(client(), settings(), docs, index=index)
    return IngestResponse(
        indexed=success, errors=errors, total_in_index=index_count(client(), settings(), index)
    )


@router.post("/search", response_model=DenseSearchResponse)
def search(req: DenseSearchRequest) -> DenseSearchResponse:
    """Dense semantic (kNN) search via the `neural` query — runs the model at query time."""
    try:
        return run_dense_search(client(), settings(), req.query, req.size)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/search", response_model=DenseSearchResponse)
def search_get(q: str, size: int = 5) -> DenseSearchResponse:
    """Convenience GET variant, e.g. /semantic/search?q=puppy"""
    return search(DenseSearchRequest(query=q, size=size))
