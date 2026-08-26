"""Local CLIP embedding (the only place the model actually runs).

CLIP has two encoders sharing one vector space: images -> vectors and text -> vectors,
positioned so an image and a matching description land close together. We use it to embed
the Flickr images at seed time and the query text at search time.

`sentence-transformers` (and its heavy `torch` dependency) is imported LAZILY inside the
functions, so the base app boots fine without the `multimodal` extra installed — you only
need it when you actually call a multimodal endpoint. The model is loaded once and cached.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from app.config import Settings

# HF's newer "Xet" transfer backend fails behind some corporate proxies (the plain
# HTTPS metadata calls succeed, but the Xet weight fetch errors out). Force the classic
# HTTPS download path so the first-run model fetch works on locked-down networks.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

_INSTALL_HINT = (
    "The multimodal extra is not installed. Run `make sync-multimodal` "
    "(or `uv sync --extra multimodal`)."
)


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    """Load CLIP once (downloads weights on first use, then cached on disk)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # extra not installed
        raise RuntimeError(_INSTALL_HINT) from exc
    return SentenceTransformer(model_name)


def embed_text(settings: Settings, query: str) -> list[float]:
    """Embed a text query into a unit-length CLIP vector (cosine-ready)."""
    model = _load_model(settings.clip_model_name)
    vec = model.encode([query], normalize_embeddings=True)[0]
    return vec.tolist()


def embed_image_bytes(settings: Settings, data: bytes) -> list[float]:
    """Embed a single in-memory image (e.g. an upload) into a unit-length CLIP vector."""
    import io

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc
    model = _load_model(settings.clip_model_name)
    image = Image.open(io.BytesIO(data)).convert("RGB")
    vec = model.encode([image], normalize_embeddings=True)[0]
    return vec.tolist()


def embed_images(settings: Settings, paths: list[Path]) -> list[list[float]]:
    """Embed image files into unit-length CLIP vectors (same space as the text).

    Processed in chunks so we never hold all decoded images in memory at once.
    """
    model = _load_model(settings.clip_model_name)
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc

    out: list[list[float]] = []
    chunk = 64
    for start in range(0, len(paths), chunk):
        batch = paths[start : start + chunk]
        images = [Image.open(p).convert("RGB") for p in batch]
        vecs = model.encode(
            images, normalize_embeddings=True, batch_size=32, show_progress_bar=True
        )
        out.extend(v.tolist() for v in vecs)
    return out
