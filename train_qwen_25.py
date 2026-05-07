# ─────────────────────────────────────────────────────────────────────────────
# train_qwen_25.py — ASL fine-tuning pipeline (Qwen2.5-VL-3B + Unsloth QLoRA)
# ─────────────────────────────────────────────────────────────────────────────
#
# USAGE
# ─────
#   Full run (2 epochs, steps auto-computed from manifest size):
#     python train_qwen_25.py --manifest-path datasets/processed/manifest.jsonl
#
#   Custom epoch count:
#     python train_qwen_25.py --manifest-path ... --epochs 3
#
#   With validation set (enables early stopping):
#     python train_qwen_25.py --manifest-path datasets/processed/train.jsonl \
#                             --val-manifest  datasets/processed/val.jsonl   \
#                             --epochs 2
#
#   Quick test (override computed steps, e.g. 300 steps only):
#     python train_qwen_25.py --manifest-path ... --max-steps 300 --output-dir my_output/
#
#   Resume from a checkpoint (continues training from saved state):
#     python train_qwen_25.py --resume asl_lora_output/checkpoint-870
#
#   Resume with custom output directory:
#     python train_qwen_25.py --resume checkpoint-870 --output-dir my_resumed_run/
#
#   Resume and extend training steps:
#     python train_qwen_25.py --resume asl_lora_output/checkpoint-870 --max-steps 2000
#
# STEP CALCULATION
# ────────────────
#   effective_batch_size = per_device_batch_size(EFFECTIVE_BATCH) × gradient_accumulation(1) = EFFECTIVE_BATCH
#   steps_per_epoch      = ceil(num_samples / EFFECTIVE_BATCH)
#   max_steps            = steps_per_epoch × num_epochs
#
#   Example: 30,000 samples → 7,500 steps/epoch × 2 epochs = 15,000 steps
#   Pass --max-steps to override (e.g. for a smoke test).
#
# OUTPUT DIRECTORIES
# ──────────────────
#   Auto-generated when --output-dir is omitted:
#     runs/run_YYYYMMDD_HHMMSS_ep2       ← epoch-based run
#     runs/run_YYYYMMDD_HHMMSS_steps300  ← max-steps override
#
# EARLY STOPPING
# ──────────────
#   Removed — with only 5 epochs, early stopping can cut training short.
#   Instead, load_best_model_at_end=True ensures the best checkpoint
#   (by eval_loss) is restored at the end of training.
#
# TOKEN BUDGET (MAX_PIXELS=1200×28×28, max_seq_length auto-computed from --frames)
# ───────────────────────────────────────────────────────────────────
#   Qwen2.5-VL uses 28×28 pixel patches (vs Qwen3-VL's 32×32).
#   Native 1280×720: ceil(1280/28)×ceil(720/28) = 46×26 = 1,196 tokens/frame
#   n_frames × 1,196 + 75 text overhead × 1.3 margin → max_seq_length
#
#   MAX_PIXELS = 1200×28×28 = 940,800 > native 921,600 → no downscaling needed.
#
#   Compute manually:      compute_max_seq_length(n_frames)
#   Or use --frames flag:   python train_qwen_25.py --frames 16 --epochs 3
#
#   ASL requires fine detail (handshapes, finger configurations, facial grammar)
#   that are lost when downscaling. Full resolution preserves these.
#
# PROCESSOR PIXEL CAP — transformers 5.x compatibility note
# ──────────────────────────────────────────────────────────
#   Qwen2VLImageProcessorFast (transformers ≥5.0) stores the pixel limits
#   inside self.size["longest_edge"] / self.size["shortest_edge"].
#   The max_pixels property has no setter → AttributeError if assigned.
#   Passing as from_pretrained kwargs → TypeError (model __init__ rejects them).
#   Correct approach: mutate the size dict directly after loading:
#     tokenizer.image_processor.size["longest_edge"] = MAX_PIXELS
#     tokenizer.image_processor.size["shortest_edge"] = 4 * 28 * 28
# ─────────────────────────────────────────────────────────────────────────────

"""ASL fine-tuning pipeline using Unsloth FastModel + QLoRA (Qwen2.5-VL-3B)."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_PATH = Path("datasets/processed/manifest.jsonl")
OUTPUT_DIR = "asl_lora_output_qwen25"
MODEL_NAME = "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit"
PROMPT = "Translate this American Sign Language video into English text."
PATCH_SIZE = 28  # Qwen2.5-VL uses 28×28 pixel patches
DEFAULT_N_FRAMES = 8
FRAME_W = 1280
FRAME_H = 720
TEXT_OVERHEAD = 75
SEQ_LENGTH_MARGIN = 1.3  # 30% safety headroom above exact token count
MAX_PIXELS = 1200 * 28 * 28  # 940,800 pixels — preserves native 1280×720 (921,600) without downscaling


def compute_max_seq_length(n_frames: int) -> int:
    """Compute max_seq_length for a given frame count at native resolution.

    Qwen2.5-VL uses 28×28 patches: 1280×720 → ceil(1280/28)×ceil(720/28) = 46×26 = 1,196 tokens/frame.
    Result includes 30% margin for safety.
    """
    tokens_per_frame = math.ceil(FRAME_W / PATCH_SIZE) * math.ceil(FRAME_H / PATCH_SIZE)
    total = n_frames * tokens_per_frame + TEXT_OVERHEAD
    return int(total * SEQ_LENGTH_MARGIN)


# Export for backward compat — inference.py and tests still import this.
MAX_SEQ_LENGTH = compute_max_seq_length(DEFAULT_N_FRAMES)


def _record_to_sample(record: dict) -> dict:
    """Convert a manifest record to ChatML with image file paths.

    Stores paths as strings so images are opened lazily by the data collator
    (via process_vision_info), avoiding loading thousands of PIL Images into RAM.
    """
    user_content = [{"type": "image", "image": fp} for fp in record["frame_paths"]]
    user_content.append({"type": "text", "text": PROMPT})
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": record["sentence"]},
        ]
    }


def _count_manifest_samples(manifest_path: Path | str) -> int:
    """Count non-empty lines in a manifest JSONL file (one sample per line)."""
    count = 0
    with open(manifest_path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def load_training_dataset(manifest_path: Path | str = MANIFEST_PATH) -> list[dict]:
    """Load manifest.jsonl and return a list of ChatML samples.

    Returns a plain list (not HF Dataset) because PIL Images can't be
    serialized by PyArrow. SFTTrainer accepts plain lists directly.
    """
    records = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return [_record_to_sample(r) for r in records]


def train(
    manifest_path: Path | str = MANIFEST_PATH,
    val_manifest_path: Path | str | None = None,
    num_epochs: int = 2,
    max_steps: int | None = None,
    output_dir: str = OUTPUT_DIR,
    resume_from_checkpoint: str | None = None,
    n_frames: int = DEFAULT_N_FRAMES,
) -> None:
    """Fine-tune Qwen2.5-VL-3B-Instruct on ASL frames with QLoRA.

    max_steps is auto-computed from num_epochs and the manifest size when not
    explicitly provided.  Pass max_steps to override (e.g. for a quick test).
    n_frames controls the expected number of frames per sample; max_seq_length
    is computed from it automatically.
    """
    import torch
    from unsloth import FastModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig, SFTTrainer

    # 1. Load model
    max_seq_length = compute_max_seq_length(n_frames)
    print(f"Computed max_seq_length={max_seq_length} for {n_frames} frames "
          f"(patch={PATCH_SIZE}×{PATCH_SIZE}, native={FRAME_W}×{FRAME_H})")
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    # Cap image resolution so n_frames of 1280×720 stay within max_seq_length.
    # Native 1280×720: 46×26 = 1,196 tokens/frame × n_frames + 75 text × 1.3 margin.
    # Qwen2.5-VL uses 28×28 patches.
    tokenizer.image_processor.size["longest_edge"] = MAX_PIXELS
    tokenizer.image_processor.size["shortest_edge"] = 4 * 28 * 28

    # 2. Add LoRA adapters
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        target_modules="all-linear",
    )

    FastModel.for_training(model)

    # 3. Load datasets
    train_dataset = load_training_dataset(manifest_path)
    val_dataset = (
        load_training_dataset(val_manifest_path) if val_manifest_path else None
    )

    # Compute max_steps from epochs if not explicitly overridden.
    EFFECTIVE_BATCH = 48
    num_samples = _count_manifest_samples(manifest_path)
    steps_per_epoch = math.ceil(num_samples / EFFECTIVE_BATCH)
    computed_steps = steps_per_epoch * num_epochs
    max_steps = max_steps if max_steps is not None else computed_steps
    print(
        f"Training: {num_samples} samples, {num_epochs} epoch(s) "
        f"→ {steps_per_epoch} steps/epoch = {computed_steps} computed steps "
        f"(max_steps={max_steps})"
    )

    # 4. Train

    # Eval args are only added when a val set is provided.
    eval_args = (
        dict(
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        if val_dataset is not None
        else dict(
            save_strategy="steps",
            save_steps=500,
        )
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        args=SFTConfig(
            per_device_train_batch_size=EFFECTIVE_BATCH,
            gradient_accumulation_steps=1,
            warmup_steps=max(1, int(max_steps * 0.1)),
            max_steps=max_steps,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=output_dir,
            report_to="tensorboard",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=max_seq_length,
            **eval_args,
        ),
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 5. Save LoRA adapters
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"LoRA adapters saved to {output_dir}")


def cli() -> None:
    """CLI entry point for train_qwen_25.py — parses args and calls train()."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-VL-3B on ASL frames with QLoRA"
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=MANIFEST_PATH,
        help="Path to train manifest.jsonl",
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=None,
        help="Path to val manifest.jsonl (enables early stopping when provided)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs (default: 2); steps are auto-computed from manifest size",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override computed steps (e.g. 300 for a quick test); takes precedence over --epochs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for LoRA adapters (auto-generated when omitted)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from a checkpoint folder (e.g. asl_lora_output/checkpoint-870)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_N_FRAMES,
        help=f"Frames per sample used for max_seq_length calculation (default: {DEFAULT_N_FRAMES})",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or _make_run_dir(args.epochs, args.max_steps)
    train(
        manifest_path=args.manifest_path,
        val_manifest_path=args.val_manifest,
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        output_dir=output_dir,
        resume_from_checkpoint=args.resume,
        n_frames=args.frames,
    )


def _make_run_dir(num_epochs: int, max_steps: int | None) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"steps{max_steps}" if max_steps is not None else f"ep{num_epochs}"
    project_root = Path(__file__).parent
    return str(project_root / "runs" / f"run_{ts}_{suffix}")


if __name__ == "__main__":
    cli()
