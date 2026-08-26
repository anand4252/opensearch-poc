"""CLI to bootstrap the multimodal POC (create the kNN image index).

Thin wrapper around `app.multimodal.bootstrap.run_bootstrap` (same logic the
`/multimodal/bootstrap` endpoint uses).

Run:  python -m app.multimodal.cli
      python -m app.multimodal.cli --recreate-index
"""

from __future__ import annotations

import argparse

from app.config import get_settings
from app.multimodal.bootstrap import run_bootstrap
from app.opensearch_client import build_client


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap OpenSearch multimodal (CLIP) index")
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="Drop and recreate the multimodal index if it already exists.",
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

    print("\nMultimodal bootstrap complete.")
    print(f"  index      : {result['index']} (created: {result['index_created']})")
    print(f"  dimension  : {result['dimension']}")
    print("\nNext:  copy the images, then  make run  and  make seed-multimodal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
