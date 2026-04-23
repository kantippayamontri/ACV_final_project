import argparse
from pathlib import Path
from preprocess.extract import build_manifest


def run_preprocess(n_frames: int = 8) -> None:
    tsv = Path("datasets/raw/val_rgb_front_clips/how2sign_realigned_val.csv")
    videos = Path("datasets/raw/val_rgb_front_clips/raw_videos")
    out = Path("datasets/processed")
    build_manifest(tsv, videos, out, n_frames=n_frames)


def main() -> None:
    parser = argparse.ArgumentParser(description="ASL fine-tuning pipeline")
    parser.add_argument("--preprocess", action="store_true", help="Extract frames and build manifest.jsonl")
    parser.add_argument("--n-frames", type=int, default=8, help="Frames to extract per clip (default: 8)")
    args = parser.parse_args()

    if args.preprocess:
        run_preprocess(n_frames=args.n_frames)
        return

    print("No action specified. Use --preprocess to extract frames.")


if __name__ == "__main__":
    main()
