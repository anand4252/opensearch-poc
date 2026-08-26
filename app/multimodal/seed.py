"""Seed the multimodal index: embed the subset images with CLIP and index the vectors.

Reuses the same subset as every other chapter: `flickr_docs.json` gives the image names
(and captions, kept only for display). We load each image file from `flickr_images_dir`,
embed the pixels with CLIP, and index `{name, combined_text, image_vector}`.
"""

from __future__ import annotations

import json

from app.config import Settings
from app.dataset import resolve
from app.ingest import bulk_index, index_count
from app.multimodal.clip_model import embed_images


def seed_images(client, settings: Settings) -> dict:
    data_file = resolve(settings.seed_data_file)
    if not data_file.exists():
        raise FileNotFoundError(
            f"Seed dataset not found: {data_file}. Build it first: "
            "run `make prepare` or POST /dataset/prepare."
        )
    docs = json.loads(data_file.read_text())

    images_dir = resolve(settings.flickr_images_dir)
    present_docs, paths, missing = [], [], 0
    for d in docs:
        img = images_dir / d["name"]
        if img.exists():
            present_docs.append(d)
            paths.append(img)
        else:
            missing += 1

    if not paths:
        raise FileNotFoundError(
            f"No image files found in {images_dir}. Copy the subset first: "
            "POST /multimodal/copy-images with the images folder, or "
            "`make copy-images IMAGES_SRC=/path/to/flickr30k_images/flickr30k_images`."
        )

    vectors = embed_images(settings, paths)
    index_docs = [
        {
            "name": d["name"],
            settings.text_field: d.get("combined_text", ""),
            settings.image_vector_field: vec,
        }
        for d, vec in zip(present_docs, vectors, strict=True)
    ]
    ok, errs = bulk_index(client, settings, index_docs, index=settings.multimodal_index_name)
    return {
        "indexed": ok,
        "errors": errs,
        "missing_images": missing,
        "total_in_index": index_count(client, settings, settings.multimodal_index_name),
    }
