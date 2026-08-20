"""Bootstrap a local OpenSearch cluster for hybrid (BM25 + neural sparse) search.

Thin CLI wrapper around `app.bootstrap.run_bootstrap` (the same logic the
`/admin/bootstrap` endpoint uses). See app/bootstrap.py for the step-by-step detail.

Run:  uv run python scripts/bootstrap_opensearch.py
      uv run python scripts/bootstrap_opensearch.py --recreate-index   # drop & recreate index
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script: `python scripts/bootstrap_opensearch.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.bootstrap import run_bootstrap  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.opensearch_client import build_client  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap OpenSearch hybrid sparse search")
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="Drop and recreate the index if it already exists.",
    )
    args = parser.parse_args()

    settings = get_settings()
    client = build_client(settings)

    try:
        health = client.cluster.health()
    except Exception as exc:  # pragma: no cover - connection error path
        print(f"ERROR: cannot reach OpenSearch at {settings.opensearch_url}: {exc}")
        print("Is the stack up?  make up   (then wait for it to become healthy)")
        return 1
    print(f"Connected. Cluster '{health['cluster_name']}' status: {health['status']}")

    result = run_bootstrap(client, settings, recreate_index=args.recreate_index)

    print("\nBootstrap complete.")
    print(f"  model_id        : {result['model_id']}")
    print(f"  model reused    : {result['model_reused']}")
    print(f"  index           : {settings.index_name} (created: {result['index_created']})")
    print(f"  ingest pipeline : {result['ingest_pipeline']}")
    print(f"  search pipeline : {result['search_pipeline']}")
    print("\nNext:  make run   then   make seed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
