# Usage:
#   python train.py --max-steps 300                          ← final project (recommended)
#   python train.py --max-steps 60 --output-dir my_output/   ← quick test

"""ASL fine-tuning pipeline using Unsloth FastVisionModel + QLoRA."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


MANIFEST_PATH = Path("datasets/processed/manifest.jsonl")
OUTPUT_DIR = "asl_lora_output"
MODEL_NAME = "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"
PROMPT = "Translate this American Sign Language video into English text."
MAX_SEQ_LENGTH = 4096  # 8 frames × ~270 tokens/frame (capped) + text overhead fits in 4096
MAX_PIXELS = 512 * 28 * 28  # cap per-frame to ~270 tokens; 8 frames ≈ 2160 visual tokens


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
    max_steps: int = 60,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """Fine-tune Qwen3 VL-2B-Instruct on ASL frames with QLoRA."""
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
    tokenizer.image_processor.max_pixels = MAX_PIXELS
    tokenizer.image_processor.min_pixels = 4 * 28 * 28
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

    # 3. Load dataset
    dataset = load_training_dataset(manifest_path)

    # 4. Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
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
        ),
    )
    trainer.train()

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
                        help="Path to manifest.jsonl")
    parser.add_argument("--max-steps", type=int, default=60,
                        help="Training steps (default: 60)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for LoRA adapters (auto-generated when omitted)")
    args = parser.parse_args()
    output_dir = args.output_dir or _make_run_dir(args.max_steps)
    train(
        manifest_path=args.manifest_path,
        max_steps=args.max_steps,
        output_dir=output_dir,
    )


def _make_run_dir(max_steps: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = f"runs/run_{ts}_steps{max_steps}"
    return name


if __name__ == "__main__":
    cli()
