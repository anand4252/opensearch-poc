"""Shared FastAPI request dependencies used by every technique router.

Kept separate from `app.main` so the per-technique routers (`app/sparse/routes.py`,
`app/semantic/routes.py`) can import these without importing the app itself.
"""

import json
from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException

from app.config import Settings, get_settings
from app.opensearch_client import build_client

REPO_ROOT = Path(__file__).resolve().parent.parent


@lru_cache
def settings() -> Settings:
    return get_settings()


@lru_cache
def client():
    return build_client(settings())


def seed_file(s: Settings) -> Path:
    """Resolve SEED_DATA_FILE against the repo root."""
    p = Path(s.seed_data_file).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def load_seed_docs() -> list[dict]:
    """Load the prepared dataset, or 404 with a hint to build it first.

    Shared by both seed endpoints — they only differ in which index they write to.
    """
    data_file = seed_file(settings())
    if not data_file.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Seed dataset not found: {data_file}. Build it first: "
                "run `make prepare` or POST /dataset/prepare."
            ),
        )
    return json.loads(data_file.read_text())
