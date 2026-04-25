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
    args = parser.parse_args()

    from preprocess.extract import build_manifest

    build_manifest(args.tsv, args.videos, args.out, n_frames=args.n_frames)


if __name__ == "__main__":
    main()
