"""Load preprocessed manifest.jsonl as a HuggingFace-compatible dataset."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

PROMPT = "Translate this American Sign Language video into English text."


def format_sample(record: dict) -> dict:
    """Convert a manifest record to the ChatML messages format expected by Unsloth."""
    user_content = [{"type": "image", "image": fp} for fp in record["frame_paths"]]
    user_content.append({"type": "text", "text": PROMPT})
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": record["sentence"]},
        ]
    }


def load_manifest_dataset(manifest_path: Path) -> Dataset:
    """Read manifest.jsonl and return a HuggingFace Dataset of formatted samples."""
    records = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    formatted = [format_sample(r) for r in records]
    flat = [{"messages": json.dumps(s["messages"])} for s in formatted]
    return Dataset.from_list(flat)
