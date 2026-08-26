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
    from app.multimodal.search import run_multimodal_search

    resp = run_multimodal_search(client, settings, "a dog on the beach", size=5)
    assert resp.query
    assert len(resp.hits) > 0
    # results carry the image filename + its caption (for display)
    assert resp.hits[0].name.endswith(".jpg")
