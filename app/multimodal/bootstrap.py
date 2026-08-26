"""Bootstrap the multimodal POC — just a kNN index.

Much simpler than the sparse/dense bootstraps: there is NO model to deploy and NO ingest
pipeline, because CLIP runs in our app and we index finished vectors directly. All this
does is create a kNN index shaped to hold CLIP's 512-dim image vectors.
"""

from __future__ import annotations


def create_index(client, settings, recreate: bool) -> bool:
    """Create the multimodal kNN index. Returns True if (re)created, False if left as-is."""
    print(f"\n[1/1] Creating multimodal index '{settings.multimodal_index_name}'...")
    if client.indices.exists(index=settings.multimodal_index_name):
        if recreate:
            print("      exists -> deleting (recreate_index).")
            client.indices.delete(index=settings.multimodal_index_name)
        else:
            print("      already exists -> skipping (use recreate_index to rebuild).")
            return False

    client.indices.create(
        index=settings.multimodal_index_name,
        body={
            "settings": {
                "index.knn": True,
                # Single node: 1 shard / 0 replicas keeps cluster health green.
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "mappings": {
                "properties": {
                    "name": {"type": "text"},
                    settings.text_field: {"type": "text"},  # caption kept for display only
                    settings.image_vector_field: {
                        "type": "knn_vector",
                        "dimension": settings.clip_dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": settings.clip_space_type,
                            "engine": settings.clip_engine,
                        },
                    },
                }
            },
        },
    )
    print("      done.")
    return True


def run_bootstrap(client, settings, *, recreate_index: bool = False) -> dict:
    """Create the kNN index and return a small summary."""
    index_created = create_index(client, settings, recreate=recreate_index)
    return {
        "index": settings.multimodal_index_name,
        "index_created": index_created,
        "dimension": settings.clip_dimension,
    }
