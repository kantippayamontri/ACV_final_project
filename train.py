# ─────────────────────────────────────────────────────────────────────────────
# train.py — ASL fine-tuning pipeline (Qwen3-VL-2B + Unsloth QLoRA)
# ─────────────────────────────────────────────────────────────────────────────
#
# USAGE
# ─────
#   Full run (2 epochs, steps auto-computed from manifest size):
#     python train.py --manifest-path datasets/processed/manifest.jsonl
#
#   Custom epoch count:
#     python train.py --manifest-path ... --epochs 3
#
#   With validation set (enables early stopping):
#     python train.py --manifest-path datasets/processed/train.jsonl \
#                     --val-manifest  datasets/processed/val.jsonl   \
#                     --epochs 2
#
#   Quick test (override computed steps, e.g. 300 steps only):
#     python train.py --manifest-path ... --max-steps 300 --output-dir my_output/
#
#   Resume from a checkpoint (continues training from saved state):
#     python train.py --resume asl_lora_output/checkpoint-870
#
#   Resume with custom output directory:
#     python train.py --resume checkpoint-870 --output-dir my_resumed_run/
#
#   Resume and extend training steps:
#     python train.py --resume asl_lora_output/checkpoint-870 --max-steps 2000
#
# STEP CALCULATION
# ────────────────
#   effective_batch_size = per_device_batch_size(1) × gradient_accumulation(4) = 4
#   steps_per_epoch      = ceil(num_samples / 4)
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
#   Activated automatically when --val-manifest is provided.
#   Evaluates every 500 steps; stops if val_loss does not improve for
#   3 consecutive evals (patience=3); restores best checkpoint at end.
#
# TOKEN BUDGET (why MAX_SEQ_LENGTH=4096 and MAX_PIXELS=512×28×28)
# ────────────────────────────────────────────────────────────────
#   Without MAX_PIXELS cap: 1280×720 frames → 1,125 tokens/frame × 8 = 9,000
#   tokens per sample — far exceeds any reasonable seq length and causes OOM.
#
#   With MAX_PIXELS = 512×28×28 = 401,408: processor resizes 1280×720 → 840×448
#   → 480 tokens/frame × 8 frames = 3,840 visual tokens.
#
#   Full sample budget:
#     3,840  visual tokens (8 frames)
#   +    30  chat template overhead
#   +    15  prompt text
#   +    30  ground-truth answer (worst case)
#   = ~3,915  total  <  MAX_SEQ_LENGTH=4096  ✓ (181 tokens headroom)
#
#   See TRAIN_CONCEPT_SUMMARY.md for full derivation.
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

"""ASL fine-tuning pipeline using Unsloth FastVisionModel + QLoRA."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path


MANIFEST_PATH = Path("datasets/processed/manifest.jsonl")
OUTPUT_DIR = "asl_lora_output"
MODEL_NAME = "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"
PROMPT = "Translate this American Sign Language video into English text."
MAX_SEQ_LENGTH = 5120 * 2  # 8 frames × ~270 tokens/frame (capped) + text overhead fits in 4096
MAX_PIXELS = 512 * 32 * 32  # cap per-frame to ~270 tokens; 8 frames ≈ 2160 visual tokens


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
    """Load manifest.jsonl and return a list of ChatML samples with PIL Images.

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
) -> None:
    """Fine-tune Qwen3 VL-2B-Instruct on ASL frames with QLoRA.

    max_steps is auto-computed from num_epochs and the manifest size when not
    explicitly provided.  Pass max_steps to override (e.g. for a quick test).
    """
    import torch
    from unsloth import FastVisionModel # allow to load and fine-tune VLLM using unsloth utilities
    from unsloth.trainer import UnslothVisionDataCollator # use to prepare batches of VLLM data (image, text) for training, handling the format and batching required by the model during fine-tuning
    from trl import SFTTrainer, SFTConfig # SFTTrainer = class for supervised fine-tuning (SFT), SFTConfig = use to config training parameters like batch size, steps, optimizers

    # 1. Load model
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    # Cap image resolution so 8 frames of 1280×720 stay within MAX_SEQ_LENGTH.
    # Without this, 1280×720 = 1125 tokens/frame × 8 = 9000 tokens >> 4096.
    # The processor stores these as size["longest_edge"] / size["shortest_edge"].
    # Setting via the property raises AttributeError in transformers 5.x (no setter);
    # passing as from_pretrained kwargs fails because the model __init__ rejects them.
    # Direct dict mutation is the only working approach.
    tokenizer.image_processor.size["longest_edge"] = MAX_PIXELS
    tokenizer.image_processor.size["shortest_edge"] = 4 * 32 * 32
    # 4-bit: Uses the least memory (about 4× smaller than 16-bit), fastest, but may lose some accuracy. Enables training very large models on consumer GPUs. Used for QLoRA.
    # 8-bit: Uses more memory than 4-bit but less than 16-bit. Good balance between efficiency and accuracy, with minimal quality loss.
    # 16-bit (fp16/bf16): Standard for most training, highest accuracy, but uses the most memory and compute. Needed for full-precision tasks.
    # In summary: Lower bits = less memory, faster, but potentially less accurate. 4-bit is most efficient, 16-bit is most precise.

    # 2. Add LoRA adapters
    model = FastVisionModel.get_peft_model( # parameter-efficient fine-tuning(PEFT)
        model,
        finetune_vision_layers=True, # enable LoRA adapters on vision encoder layers -> use for process images/frames -> learn ASL vision features
        finetune_language_layers=True, # enable LoRA adapters on language encoder layers -> use for process generate text -> learn ASL translations
        finetune_attention_modules=True, # enable LoRA adapters on attention module -> Q,K,V projections -> learn new attention pattern for ASL
        finetune_mlp_modules=True, # enable LoRA adapters on MLP(feed-forward) modules -> learn new transformations for ASL data.
        r=16, #
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407, # random seed for reproducible
        use_rslora=False,
        target_modules="all-linear", # Apply LoRA adapters to all linear layers in the model, not just specific ones. Maximizes what the adapters can learn.
    )

    FastVisionModel.for_training(model)
    # 1) Enables gradient checkpointing — saves memory by recomputing activations during the backward pass instead of storing them.
    # 2) Sets the model to train mode — activates dropout, batch norm updates, etc.
    # 3) Ensures LoRA adapters are trainable — makes sure only the LoRA parameters have gradients enabled, while the frozen base model weights stay untrainable.

    # 3. Load datasets
    train_dataset = load_training_dataset(manifest_path)
    val_dataset = load_training_dataset(val_manifest_path) if val_manifest_path else None

    # Compute max_steps from epochs if not explicitly overridden.
    # effective batch size = per_device_batch(1) × grad_accum(4) = 4
    EFFECTIVE_BATCH = 4
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
    from transformers import EarlyStoppingCallback

    # Eval args are only added when a val set is provided.
    # load_best_model_at_end requires save_strategy == eval_strategy.
    eval_args = dict(
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    ) if val_dataset is not None else dict(
        save_strategy="steps",
        save_steps=500,
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)] if val_dataset is not None else []

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,           # None when no val set → SFTTrainer ignores it
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        callbacks=callbacks,
        args=SFTConfig(
            per_device_train_batch_size=EFFECTIVE_BATCH,
            gradient_accumulation_steps=1,
            warmup_steps=5,
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
            report_to="none",
            # Required for vision fine-tuning:
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=MAX_SEQ_LENGTH,
            **eval_args,
        ),
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 5. Save LoRA adapters
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"LoRA adapters saved to {output_dir}")


def cli() -> None:
    """CLI entry point for train.py — parses args and calls train()."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-VL on ASL frames with QLoRA"
    )
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH,
                        help="Path to train manifest.jsonl")
    parser.add_argument("--val-manifest", type=Path, default=None,
                        help="Path to val manifest.jsonl (enables early stopping when provided)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs (default: 2); steps are auto-computed from manifest size")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override computed steps (e.g. 300 for a quick test); takes precedence over --epochs")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for LoRA adapters (auto-generated when omitted)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from a checkpoint folder (e.g. asl_lora_output/checkpoint-870)")
    args = parser.parse_args()
    output_dir = args.output_dir or _make_run_dir(args.epochs, args.max_steps)
    train(
        manifest_path=args.manifest_path,
        val_manifest_path=args.val_manifest,
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        output_dir=output_dir,
        resume_from_checkpoint=args.resume,
    )


def _make_run_dir(num_epochs: int, max_steps: int | None) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"steps{max_steps}" if max_steps is not None else f"ep{num_epochs}"
    return f"runs/run_{ts}_{suffix}"


if __name__ == "__main__":
    cli()
