"""ASL fine-tuning pipeline using Unsloth FastVisionModel + QLoRA."""

from __future__ import annotations

import json
from pathlib import Path


MANIFEST_PATH = Path("datasets/processed/manifest.jsonl")
OUTPUT_DIR = "asl_lora_output"
MODEL_NAME = "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"
PROMPT = "Translate this American Sign Language video into English text."
MAX_SEQ_LENGTH = 2048


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
    """Fine-tune Qwen2.5-VL-2B-Instruct on ASL frames with QLoRA."""
    import torch
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    # 1. Load model
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    # 2. Add LoRA adapters
    model = FastVisionModel.get_peft_model(
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

    FastVisionModel.for_training(model)

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
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Output directory for LoRA adapters")
    args = parser.parse_args()
    train(
        manifest_path=args.manifest_path,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    cli()
