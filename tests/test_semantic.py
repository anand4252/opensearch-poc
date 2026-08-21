"""Dense semantic smoke tests. Require a bootstrapped+seeded dense index; skipped otherwise."""

import pytest

from app.config import get_settings
from app.opensearch_client import build_client
from app.semantic.search import run_dense_search


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
    if not c.indices.exists(index=settings.dense_index_name):
        pytest.skip(
            f"Dense index '{settings.dense_index_name}' missing — "
            "run bootstrap-semantic + seed-semantic first"
        )
    return c


def test_dense_search_returns_hits(client, settings):
    # "man" is common in the captions, so dense retrieval should return hits.
    resp = run_dense_search(client, settings, "man", size=5)
    assert resp.query == "man"
    assert resp.model_id  # a deployed dense model id was resolved
    assert len(resp.hits) > 0


def test_dense_semantic_matches_absent_word(client, settings):
    # "puppy" never appears literally in the captions; dense semantic should still
    # surface dog photos (the "money shot" — semantic, not string, matching).
    resp = run_dense_search(client, settings, "puppy", size=5)
    assert len(resp.hits) > 0
    assert any("dog" in h.combined_text.lower() for h in resp.hits)
