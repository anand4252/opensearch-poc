"""FastAPI app exposing health, ingest, and hybrid-search endpoints."""

import json
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.config import Settings, get_settings
from app.dataset import build_dataset
from app.ingest import bulk_index, index_count
from app.models import (
    IngestRequest,
    IngestResponse,
    PrepareResponse,
    SearchMode,
    SearchRequest,
    SearchResponse,
)
from app.opensearch_client import build_client
from app.search import run_search

REPO_ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="OpenSearch Hybrid Sparse Search", version="0.1.0")


def _seed_file(s: Settings) -> Path:
    p = Path(s.seed_data_file).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


@lru_cache
def settings() -> Settings:
    return get_settings()


@lru_cache
def client():
    return build_client(settings())


@app.get("/health")
def health():
    try:
        h = client().cluster.health()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"OpenSearch unreachable: {exc}") from exc
    return {"status": h["status"], "cluster_name": h["cluster_name"]}


@app.post("/documents", response_model=IngestResponse)
def add_documents(req: IngestRequest) -> IngestResponse:
    docs = [d.model_dump() for d in req.documents]
    success, errors = bulk_index(client(), settings(), docs)
    return IngestResponse(
        indexed=success, errors=errors, total_in_index=index_count(client(), settings())
    )


@app.post("/documents/prepare", response_model=PrepareResponse)
def prepare_dataset(size: int | None = None) -> PrepareResponse:
    """Build the seed dataset (data/results.csv -> data/flickr_docs.json).

    Same transform as `make prepare`; handy for a click-through demo (prepare -> seed).
    Optional `size` overrides FLICKR_SUBSET_SIZE for this build.
    """
    try:
        result = build_dataset(settings(), size=size)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PrepareResponse(**result)


@app.post("/documents/seed", response_model=IngestResponse)
def seed_documents() -> IngestResponse:
    data_file = _seed_file(settings())
    if not data_file.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Seed dataset not found: {data_file}. Build it first: "
                "run `make prepare` or POST /documents/prepare."
            ),
        )
    docs = json.loads(data_file.read_text())
    success, errors = bulk_index(client(), settings(), docs)
    return IngestResponse(
        indexed=success, errors=errors, total_in_index=index_count(client(), settings())
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    try:
        return run_search(client(), settings(), req.query, req.mode, req.size)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/search", response_model=SearchResponse)
def search_get(q: str, mode: SearchMode = SearchMode.hybrid, size: int = 5) -> SearchResponse:
    """Convenience GET variant, e.g. /search?q=horror%20movies&mode=hybrid"""
    return search(SearchRequest(query=q, mode=mode, size=size))
