"""Smoke tests. These require a running, bootstrapped cluster and are skipped otherwise."""

import pytest

from app.config import get_settings
from app.models import SearchMode
from app.opensearch_client import build_client
from app.search import build_body, run_search


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
    if not c.indices.exists(index=settings.index_name):
        pytest.skip(f"Index '{settings.index_name}' missing — run bootstrap + seed first")
    return c


def test_build_body_hybrid_has_two_subqueries(settings):
    body = build_body(settings, "horror movies", SearchMode.hybrid, 5)
    queries = body["query"]["hybrid"]["queries"]
    assert len(queries) == 2
    assert "match" in queries[0]
    assert "neural_sparse" in queries[1]


def test_build_body_lexical_is_match(settings):
    body = build_body(settings, "spider man", SearchMode.lexical, 3)
    assert "match" in body["query"]


@pytest.mark.parametrize("mode", list(SearchMode))
def test_search_returns_hits(client, settings, mode):
    resp = run_search(client, settings, "scary horror movies", mode, size=5)
    assert resp.mode is mode
    assert len(resp.hits) > 0
