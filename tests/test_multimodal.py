"""Multimodal smoke tests. Require the multimodal extra + a seeded index; skipped otherwise."""

import pytest

from app.config import get_settings
from app.opensearch_client import build_client

pytest.importorskip("sentence_transformers", reason="multimodal extra not installed")


@pytest.fixture(scope="module")
def settings():
    return get_settings()


@pytest.fixture(scope="module")
def client(settings):
    c = build_client(settings)
    try:
        c.cluster.health(request_timeout=2)
    except Exception:
        pytest.skip("OpenSearch not reachable — skipping integration tests")
    if not c.indices.exists(index=settings.multimodal_index_name):
        pytest.skip(
            f"Multimodal index '{settings.multimodal_index_name}' missing — "
            "run copy-images + bootstrap-multimodal + seed-multimodal first"
        )
    return c


def test_text_to_image_returns_hits(client, settings):
    from app.multimodal.search import run_text_search

    resp = run_text_search(client, settings, "a dog on the beach", size=5)
    assert resp.query
    assert len(resp.hits) > 0
    # results carry the image filename + its caption (for display)
    assert resp.hits[0].name.endswith(".jpg")


def test_image_to_image_finds_itself(client, settings):
    import json

    from app.dataset import resolve
    from app.multimodal.search import run_image_search

    docs = json.loads(resolve(settings.seed_data_file).read_text())
    name = docs[0]["name"]
    img = resolve(settings.flickr_images_dir) / name
    if not img.exists():
        pytest.skip("subset images not copied to data/images")
    resp = run_image_search(client, settings, img.read_bytes(), size=3, label=name)
    # an image is its own nearest neighbor (cosine similarity 1.0)
    assert resp.hits[0].name == name
