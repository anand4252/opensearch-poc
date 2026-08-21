"""Bulk indexing helpers.

Each document passes through the index's default ingest pipeline, which runs the
sparse-encoding model. That is relatively slow per document, so we keep batches
small (mirroring the AWS setup this is modelled on).
"""

from opensearchpy import helpers

from app.config import Settings

BATCH_SIZE = 25


def bulk_index(
    client, settings: Settings, documents: list[dict], index: str | None = None
) -> tuple[int, int]:
    """Index documents into `index` (defaults to the sparse index); return (ok, errors)."""
    target = index or settings.index_name
    success_count = 0
    error_count = 0

    for start in range(0, len(documents), BATCH_SIZE):
        batch = documents[start : start + BATCH_SIZE]
        # Use the image filename (`name`) as `_id` so re-seeding overwrites the same
        # document instead of appending a duplicate (indexing is idempotent per _id).
        actions = [
            {"_index": target, "_id": doc.get("name"), "_source": doc}
            for doc in batch
        ]
        success, errors = helpers.bulk(
            client, actions, raise_on_error=False, request_timeout=300
        )
        success_count += success
        if errors:
            error_count += len(errors)

    client.indices.refresh(index=target)
    return success_count, error_count


def index_count(client, settings: Settings, index: str | None = None) -> int:
    return client.count(index=index or settings.index_name)["count"]
