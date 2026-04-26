"""ASL fine-tuning pipeline — thin wrapper over preprocess_data.py and train.py."""
from __future__ import annotations

import argparse

import preprocess_data
import train as train_module


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
        from preprocess.extract import build_manifest
        build_manifest(
            preprocess_data.DEFAULT_TSV,
            preprocess_data.DEFAULT_VIDEOS,
            preprocess_data.DEFAULT_OUT,
            n_frames=args.n_frames,
        )
    elif args.train:
        from train import _make_run_dir
        output_dir = _make_run_dir(args.max_steps)
        train_module.train(max_steps=args.max_steps, output_dir=output_dir)
    else:
        print("No action specified. Use --preprocess or --train.")


if __name__ == "__main__":
    main()
