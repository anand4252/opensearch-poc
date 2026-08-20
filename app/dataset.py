"""Build the shared search dataset from the Flickr30k captions.

The Flickr30k dataset (`data/results.csv`) has 5 human-written captions per image
(`image_name | comment_number | comment`). This module groups the 5 captions per
image, takes a deterministic subset, and writes a compact JSON dataset:

    [{ "name": "<image_name>", "combined_text": "<the 5 captions joined>" }, ...]

`name` is the image filename, which is also the join key to the image pixels used by
the (later) multimodal POC. Output is deterministic (images sorted, first N taken).

This logic is shared by both `scripts/prepare_flickr.py` (CLI) and the
`POST /documents/prepare` endpoint, so there is a single source of truth.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

from app.config import Settings

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve(path_str: str) -> Path:
    """Resolve a possibly-relative path against the repo root."""
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def group_captions(csv_path: Path) -> dict[str, list[str]]:
    """image_name -> [caption, ...]. CSV is pipe-delimited with padded whitespace."""
    captions: dict[str, list[str]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="|")
        header = next(reader, None)  # image_name | comment_number | comment
        if header is None:
            raise ValueError(f"{csv_path} is empty")
        for row in reader:
            if len(row) < 3:
                continue  # skip malformed lines
            image_name = row[0].strip()
            # join row[2:] so a stray '|' inside a caption isn't truncated
            comment = "|".join(row[2:]).strip()
            if image_name and comment:
                captions[image_name].append(comment)
    return captions


def build_docs(captions: dict[str, list[str]], size: int) -> list[dict[str, str]]:
    """Deterministic subset: sort image names, take first `size`, join captions."""
    return [
        {"name": name, "combined_text": " ".join(captions[name])}
        for name in sorted(captions)[:size]
    ]


def build_dataset(
    settings: Settings,
    *,
    size: int | None = None,
    csv_path: str | None = None,
    out_path: str | None = None,
) -> dict:
    """Read the captions CSV, write the subset dataset JSON, return a summary dict."""
    csv_p = resolve(csv_path or settings.flickr_csv)
    out_p = resolve(out_path or settings.seed_data_file)
    if not csv_p.exists():
        raise FileNotFoundError(
            f"Captions CSV not found: {csv_p}. Expected data/results.csv "
            "(committed) — see README 'Data setup'."
        )
    captions = group_captions(csv_p)
    docs = build_docs(captions, size or settings.flickr_subset_size)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(docs, indent=2, ensure_ascii=False) + "\n")
    return {
        "images_available": len(captions),
        "docs_written": len(docs),
        "output_file": str(out_p),
    }


def copy_images(docs: list[dict[str, str]], images_src: Path, images_dir: Path) -> int:
    """Copy the subset's .jpg files into `images_dir` (for the multimodal POC)."""
    if not images_src.is_dir():
        raise FileNotFoundError(f"images source dir not found: {images_src}")
    images_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for doc in docs:
        src = images_src / doc["name"]
        if src.exists():
            shutil.copy2(src, images_dir / doc["name"])
            copied += 1
        else:
            print(f"      WARN: image missing, skipped: {src}")
    return copied
