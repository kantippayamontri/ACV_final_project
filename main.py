import argparse
from pathlib import Path


def run_preprocess(n_frames: int = 8) -> None:
    from preprocess.extract import build_manifest
    tsv = Path("datasets/raw/val_rgb_front_clips/how2sign_realigned_val.csv")
    videos = Path("datasets/raw/val_rgb_front_clips/raw_videos")
    out = Path("datasets/processed")
    build_manifest(tsv, videos, out, n_frames=n_frames)


def run_train(max_steps: int = 60) -> None:
    from train import train
    train(max_steps=max_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description="ASL fine-tuning pipeline")
    parser.add_argument("--preprocess", action="store_true",
                        help="Extract frames and build manifest.jsonl")
    parser.add_argument("--n-frames", type=int, default=8,
                        help="Frames to extract per clip (default: 8)")
    parser.add_argument("--train", action="store_true",
                        help="Fine-tune Qwen2.5-VL-2B-Instruct on preprocessed data")
    parser.add_argument("--max-steps", type=int, default=60,
                        help="Training steps (default: 60)")
    args = parser.parse_args()

    if args.preprocess:
        run_preprocess(n_frames=args.n_frames)
    elif args.train:
        run_train(max_steps=args.max_steps)
    else:
        print("No action specified. Use --preprocess or --train.")


if __name__ == "__main__":
    main()
