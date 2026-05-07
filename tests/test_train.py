"""Tests for train.py — no GPU required, no real video I/O."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage


def write_manifest(path: Path, frame_dirs: list[Path], sentences: list[str]) -> None:
    with open(path, "w") as f:
        for i, (frame_dir, sentence) in enumerate(zip(frame_dirs, sentences)):
            fps = sorted(str(p) for p in frame_dir.glob("*.jpg"))
            f.write(json.dumps({"clip_name": f"clip_{i:03d}", "sentence": sentence, "frame_paths": fps}) + "\n")


def make_frames(base: Path, clip_name: str, n: int = 4) -> Path:
    d = base / clip_name
    d.mkdir(parents=True)
    for j in range(n):
        PILImage.fromarray(np.zeros((32, 32, 3), dtype="uint8")).save(d / f"frame_{j:02d}.jpg")
    return d


def test_load_training_dataset_returns_correct_length(tmp_path):
    dirs = [make_frames(tmp_path, f"clip_{i:03d}") for i in range(5)]
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, dirs, [f"sentence {i}" for i in range(5)])

    from train import load_training_dataset
    ds = load_training_dataset(manifest)
    assert len(ds) == 5


def test_load_training_dataset_sample_has_messages(tmp_path):
    d = make_frames(tmp_path, "clip_000", n=4)
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [d], ["hello world"])

    from train import load_training_dataset
    ds = load_training_dataset(manifest)
    sample = ds[0]

    assert "messages" in sample
    assert sample["messages"][0]["role"] == "user"
    assert sample["messages"][1]["role"] == "assistant"
    assert sample["messages"][1]["content"] == "hello world"


def test_load_training_dataset_images_are_paths(tmp_path):
    n_frames = 4
    d = make_frames(tmp_path, "clip_000", n=n_frames)
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [d], ["test sentence"])

    from train import load_training_dataset
    ds = load_training_dataset(manifest)
    user_content = ds[0]["messages"][0]["content"]

    image_entries = [c for c in user_content if c["type"] == "image"]
    assert len(image_entries) == n_frames
    for entry in image_entries:
        assert isinstance(entry["image"], str), f"Expected string path, got {type(entry['image'])}"
        assert Path(entry["image"]).suffix == ".jpg"


def test_max_seq_length_fits_8_frames():
    from train import MAX_SEQ_LENGTH
    # 8 frames of 1280x720 at max_pixels=512*28*28 ≈ 270 tokens/frame = ~2160
    # Plus text overhead ~65 → minimum safe value is 4096
    assert MAX_SEQ_LENGTH >= 4096, (
        f"MAX_SEQ_LENGTH={MAX_SEQ_LENGTH} is too small for 8 frames; need >= 4096"
    )


def test_max_pixels_constant_defined():
    from train import MAX_PIXELS
    expected = 1024 * 32 * 32
    assert MAX_PIXELS == expected, (
        f"MAX_PIXELS should be 1024*32*32={expected}, got {MAX_PIXELS}"
    )
