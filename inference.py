# Usage:
#   python inference.py --run-dir runs/run_20260426_143022_steps300
#   python inference.py --run-dir asl_lora_output --max-samples 20
#   python inference.py --run-dir runs/run_20260426_143022_steps300 --manifest datasets/processed/manifest.jsonl

"""ASL inference script — loads fine-tuned LoRA adapters and evaluates with BLEU."""

from __future__ import annotations

import json
from pathlib import Path

from train import MANIFEST_PATH, MODEL_NAME, MAX_SEQ_LENGTH, MAX_PIXELS, PROMPT


def load_manifest(manifest_path: Path) -> list[dict]:
    """Load manifest.jsonl and return list of records."""
    records = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_inference(
    run_dir: str,
    manifest_path: Path | str = MANIFEST_PATH,
    max_samples: int | None = None,
) -> None:
    """Load fine-tuned model and run inference on manifest, printing BLEU score."""
    import torch
    import sacrebleu
    from PIL import Image
    from unsloth import FastVisionModel

    # 1. Load base model + LoRA adapters
    print(f"Loading model from {run_dir} ...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=run_dir,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    # Apply same pixel cap as training to ensure consistent tokenization.
    tokenizer.image_processor.max_pixels = MAX_PIXELS
    tokenizer.image_processor.min_pixels = 4 * 28 * 28
    FastVisionModel.for_inference(model)

    # 2. Load manifest
    records = load_manifest(Path(manifest_path))
    if max_samples is not None:
        records = records[:max_samples]
    print(f"Running inference on {len(records)} samples ...\n")

    predictions: list[str] = []
    references: list[str] = []

    for i, record in enumerate(records):
        ground_truth = record["sentence"]

        # Load frames as PIL Images
        images = [Image.open(fp).convert("RGB") for fp in record["frame_paths"]]

        # Build messages in ChatML format (no assistant turn — we want the model to generate it)
        user_content = [{"type": "image"} for _ in images]
        user_content.append({"type": "text", "text": PROMPT})
        messages = [{"role": "user", "content": user_content}]

        # Apply chat template to get input text
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Build model inputs — pass PIL images directly to the processor
        inputs = tokenizer(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # Generate prediction
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,      # greedy — deterministic
                temperature=1.0,      # ignored when do_sample=False but required by some versions
                use_cache=True,
            )

        # Decode only the newly generated tokens (strip the input prompt)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        predictions.append(prediction)
        references.append(ground_truth)

        # Print per-sample result
        print(f"[{i+1}/{len(records)}]")
        print(f"  GT  : {ground_truth}")
        print(f"  PRED: {prediction}")
        print()

    # 3. Compute BLEU (sacrebleu expects list[str] hypotheses, list[list[str]] references)
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    exact_matches = sum(p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references))
    exact_match_pct = exact_matches / len(predictions) * 100

    print("=" * 60)
    print(f"Samples evaluated : {len(predictions)}")
    print(f"BLEU-4            : {bleu.score:.2f}")
    print(f"Exact Match       : {exact_matches}/{len(predictions)} ({exact_match_pct:.1f}%)")
    print("=" * 60)


def cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run ASL inference with BLEU evaluation")
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Path to fine-tuned LoRA adapters (e.g. runs/run_20260426_143022_steps300)"
    )
    parser.add_argument(
        "--manifest", type=Path, default=MANIFEST_PATH,
        help=f"Path to manifest.jsonl (default: {MANIFEST_PATH})"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit inference to first N samples (default: all)"
    )
    args = parser.parse_args()
    run_inference(
        run_dir=args.run_dir,
        manifest_path=args.manifest,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    cli()
