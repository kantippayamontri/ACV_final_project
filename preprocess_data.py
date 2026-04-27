# To run preprocessing directly:
#   python preprocess_data.py --tsv <path/to/val.csv> --videos <path/to/videos/> --out <output/dir> --n-frames 8 --workers 4
#
# Example with How2Sign defaults (no arguments needed):
#   python preprocess_data.py

"""Standalone preprocessing script: extract frames and build manifest.jsonl."""
from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_TSV = Path("datasets/raw/val_rgb_front_clips/how2sign_realigned_val.csv")
DEFAULT_VIDEOS = Path("datasets/raw/val_rgb_front_clips/raw_videos")
DEFAULT_OUT = Path("datasets/processed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frames from How2Sign clips and write manifest.jsonl"
    )
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV, help="Path to How2Sign TSV/CSV file")
    parser.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS, help="Directory containing MP4 files")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory for frames and manifest.jsonl")
    parser.add_argument("--n-frames", type=int, default=8, help="Frames to extract per clip (default: 8)")
    parser.add_argument("--workers", type=int, default=1, help="CPU cores/processes to use for frame extraction (default: 1)")
    parser.add_argument("--no-progress", action="store_true", help="Disable preprocessing progress bar")
    args = parser.parse_args()

    from preprocess.extract import build_manifest

    build_manifest(
        args.tsv,
        args.videos,
        args.out,
        n_frames=args.n_frames,
        show_progress=not args.no_progress,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
