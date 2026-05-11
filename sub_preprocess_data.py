#!/usr/bin/env python
# To run preprocessing with a smaller subset directly:
#   python sub_preprocess_data.py --limit 1000 --tsv <path/to/train.csv> --videos <path/to/videos/> --out <output/dir>
#
# Example with How2Sign train defaults and first 1000 videos:
#   python sub_preprocess_data.py

"""Subset preprocessing script: select the first N clips, extract frames, and build manifest.jsonl."""
from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

DEFAULT_TSV = Path("datasets/raw/train_rgb_front_clips/how2sign_realigned_train.csv")
DEFAULT_VIDEOS = Path("datasets/raw/train_rgb_front_clips/raw_videos")
DEFAULT_OUT = Path("datasets/processed_subset")
DEFAULT_LIMIT = 1000


def write_subset_tsv(tsv_path: Path, limit: int) -> Path:
    """Write the first `limit` rows of `tsv_path` to a temporary TSV file.

    The existing preprocessing pipeline expects a TSV/CSV-like file containing
    the original How2Sign columns.  This helper keeps the header unchanged and
    only limits the number of data rows passed to `build_manifest`.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")

    with open(tsv_path,encoding = "utf-8-sig" ,newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in TSV file: {tsv_path}")

        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            suffix=".tsv",
            prefix="how2sign_subset_",
            delete=False,
        )
        with temp_file:
            writer = csv.DictWriter(temp_file, fieldnames=reader.fieldnames, delimiter="\t")
            writer.writeheader()
            for index, row in enumerate(reader):
                if index >= limit:
                    break
                writer.writerow(row)

    return Path(temp_file.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frames and build manifest.jsonl for the first N How2Sign clips"
    )
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV, help="Path to How2Sign TSV/CSV file")
    parser.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS, help="Directory containing MP4 files")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory for subset frames and manifest.jsonl")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Number of rows/videos to preprocess (default: 1000)")
    parser.add_argument("--n-frames", type=int, default=8, help="Frames to extract per clip (default: 8)")
    parser.add_argument("--workers", type=int, default=1, help="CPU cores/processes to use for frame extraction (default: 1)")
    parser.add_argument("--no-progress", action="store_true", help="Disable preprocessing progress bar")
    args = parser.parse_args()

    from preprocess.extract import build_manifest

    subset_tsv = write_subset_tsv(args.tsv, args.limit)
    print(f"Using subset TSV with first {args.limit} rows: {subset_tsv}")

    build_manifest(
        subset_tsv,
        args.videos,
        args.out,
        n_frames=args.n_frames,
        show_progress=not args.no_progress,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()