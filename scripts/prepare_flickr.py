"""CLI to build the shared Flickr caption dataset (thin wrapper around app.dataset).

The transform logic lives in `app/dataset.py` so the CLI and the
`POST /documents/prepare` endpoint share a single source of truth.

Run:  uv run python scripts/prepare_flickr.py
      uv run python scripts/prepare_flickr.py --size 500
      uv run python scripts/prepare_flickr.py --copy-images \
          --images-src ~/Downloads/flickr30k_images/flickr30k_images
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a plain script: `python scripts/prepare_flickr.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.dataset import build_dataset, copy_images, resolve  # noqa: E402


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build the Flickr caption dataset")
    parser.add_argument("--size", type=int, default=settings.flickr_subset_size,
                        help="Number of images (documents) to include.")
    parser.add_argument("--csv", default=settings.flickr_csv, help="Path to the captions CSV.")
    parser.add_argument("--out", default=settings.seed_data_file, help="Output JSON dataset path.")
    parser.add_argument("--copy-images", action="store_true",
                        help="Also copy the subset's .jpg files into --images-dir.")
    parser.add_argument("--images-src", default="",
                        help="Source dir of the full Flickr images (required with --copy-images).")
    parser.add_argument("--images-dir", default=settings.flickr_images_dir,
                        help="Destination dir for the copied subset images.")
    args = parser.parse_args()

    try:
        result = build_dataset(settings, size=args.size, csv_path=args.csv, out_path=args.out)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"{result['images_available']} images available; "
          f"wrote {result['docs_written']} docs -> {result['output_file']}")

    if args.copy_images:
        if not args.images_src:
            print("ERROR: --copy-images requires --images-src <dir>")
            return 1
        docs = json.loads(Path(result["output_file"]).read_text())
        images_dir = resolve(args.images_dir)
        print(f"Copying {len(docs)} images -> {images_dir} ...")
        copied = copy_images(docs, resolve(args.images_src), images_dir)
        print(f"  copied {copied}/{len(docs)} images.")

    print("\nNext:  make seed   (with the API running)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
