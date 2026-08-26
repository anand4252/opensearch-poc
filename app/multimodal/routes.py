"""API routes for the multimodal POC (CLIP text->image search)."""

import importlib.util
import json

from fastapi import APIRouter, HTTPException, UploadFile

from app.dataset import copy_images, resolve
from app.deps import client, settings
from app.models import (
    CopyImagesRequest,
    CopyImagesResponse,
    MultimodalBootstrapResponse,
    MultimodalSearchRequest,
    MultimodalSearchResponse,
    MultimodalSeedResponse,
)
from app.multimodal.bootstrap import run_bootstrap
from app.multimodal.search import run_image_search, run_text_search
from app.multimodal.seed import seed_images

# The file-upload route needs python-multipart (ships with the `multimodal` extra).
# FastAPI errors at route-registration time if it's missing, so only register that route
# when it's available — the base app (and the dep-free by-name route) still boot without it.
_HAS_MULTIPART = (
    importlib.util.find_spec("multipart") is not None
    or importlib.util.find_spec("python_multipart") is not None
)

router = APIRouter(prefix="/multimodal", tags=["multimodal"])


@router.post("/copy-images", response_model=CopyImagesResponse)
def copy_subset_images(req: CopyImagesRequest) -> CopyImagesResponse:
    """Copy the subset's .jpg files from `images_src` into data/images (gitignored).

    Same as `make copy-images`; the click-through step before seeding. Pass the folder
    that holds the full Flickr images, e.g. ~/Downloads/flickr30k_images/flickr30k_images.
    """
    s = settings()
    data_file = resolve(s.seed_data_file)
    if not data_file.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Seed dataset not found: {data_file}. Build it first: "
                "run `make prepare` or POST /dataset/prepare."
            ),
        )
    docs = json.loads(data_file.read_text())
    images_dir = resolve(s.flickr_images_dir)
    try:
        copied = copy_images(docs, resolve(req.images_src), images_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return CopyImagesResponse(
        copied=copied,
        requested=len(docs),
        missing=len(docs) - copied,
        images_dir=str(images_dir),
    )


@router.post("/bootstrap", response_model=MultimodalBootstrapResponse)
def bootstrap(recreate_index: bool = False) -> MultimodalBootstrapResponse:
    """Create the multimodal kNN index (no model/pipeline — CLIP runs in the app)."""
    try:
        result = run_bootstrap(client(), settings(), recreate_index=recreate_index)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return MultimodalBootstrapResponse(**result)


@router.post("/seed", response_model=MultimodalSeedResponse)
def seed() -> MultimodalSeedResponse:
    """Embed the subset images with CLIP and index the vectors (slow on first call)."""
    try:
        result = seed_images(client(), settings())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return MultimodalSeedResponse(**result)


@router.post("/search", response_model=MultimodalSearchResponse)
def search(req: MultimodalSearchRequest) -> MultimodalSearchResponse:
    """Text->image search: your text is embedded with CLIP and matched to image vectors."""
    try:
        return run_text_search(client(), settings(), req.query, req.size)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/search", response_model=MultimodalSearchResponse)
def search_get(q: str, size: int = 5) -> MultimodalSearchResponse:
    """Convenience GET variant, e.g. /multimodal/search?q=a%20dog%20on%20the%20beach"""
    return search(MultimodalSearchRequest(query=q, size=size))


@router.get("/search-by-name", response_model=MultimodalSearchResponse)
def search_by_name(name: str, size: int = 5) -> MultimodalSearchResponse:
    """Image -> image using an image already in data/images (e.g. name=1000092795.jpg).

    Dep-free way to try image search: the nearest hit is the image itself, then lookalikes.
    """
    img = resolve(settings().flickr_images_dir) / name
    if not img.exists():
        raise HTTPException(status_code=404, detail=f"Image not found in data/images: {name}")
    return run_image_search(client(), settings(), img.read_bytes(), size, label=name)


if _HAS_MULTIPART:

    @router.post("/search-by-image", response_model=MultimodalSearchResponse)
    async def search_by_image(
        file: UploadFile, size: int = 5
    ) -> MultimodalSearchResponse:
        """Image -> image: upload a photo, get the visually most similar indexed images."""
        data = await file.read()
        try:
            return run_image_search(
                client(), settings(), data, size, label=file.filename or "uploaded image"
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
