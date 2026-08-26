"""FastAPI app for the OpenSearch AI-search POCs.

Thin assembler: shared endpoints (health, dataset prepare) live here; each technique's
endpoints live in its own router (`app/sparse/routes.py`, `app/semantic/routes.py`) and
are mounted below. See docs/ for the per-chapter write-ups.
"""

from fastapi import FastAPI, HTTPException

from app.dataset import build_dataset
from app.deps import client, settings
from app.models import PrepareResponse
from app.multimodal.routes import router as multimodal_router
from app.semantic.routes import router as semantic_router
from app.sparse.routes import router as sparse_router

app = FastAPI(title="OpenSearch AI Search POCs", version="0.1.0")


@app.get("/health", tags=["shared"])
def health():
    try:
        h = client().cluster.health()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"OpenSearch unreachable: {exc}") from exc
    return {"status": h["status"], "cluster_name": h["cluster_name"]}


@app.post("/dataset/prepare", response_model=PrepareResponse, tags=["shared"])
def prepare_dataset(size: int = settings().flickr_subset_size) -> PrepareResponse:
    """Build the shared seed dataset (data/results.csv -> data/flickr_docs.json).

    Same transform as `make prepare`; handy for a click-through demo (prepare -> seed).
    Both the sparse and semantic seeds read this file. `size` defaults to
    FLICKR_SUBSET_SIZE and sets how many images to build.
    """
    try:
        result = build_dataset(settings(), size=size)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PrepareResponse(**result)


app.include_router(sparse_router)
app.include_router(semantic_router)
app.include_router(multimodal_router)
