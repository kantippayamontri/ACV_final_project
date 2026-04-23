import json
import tempfile
from pathlib import Path
import pytest
from preprocess.dataset import load_manifest_dataset, format_sample


def write_manifest(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def make_fake_frames(base: Path, clip_name: str, n: int) -> list[str]:
    d = base / clip_name
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = d / f"frame_{i:02d}.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 10)  # minimal JPEG header
        paths.append(str(p))
    return paths


def test_load_manifest_dataset_length(tmp_path):
    frame_paths = make_fake_frames(tmp_path, "clip_a", 4)
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        manifest,
        [
            {
                "clip_name": "clip_a",
                "sentence": "hello",
                "frame_paths": frame_paths,
            },
            {
                "clip_name": "clip_b",
                "sentence": "world",
                "frame_paths": make_fake_frames(tmp_path, "clip_b", 4),
            },
        ],
    )
    ds = load_manifest_dataset(manifest)
    assert len(ds) == 2


def test_format_sample_structure():
    record = {
        "clip_name": "clip_a",
        "sentence": "nice to meet you",
        "frame_paths": ["/fake/frame_00.jpg", "/fake/frame_01.jpg"],
    }
    sample = format_sample(record)
    assert sample["messages"][0]["role"] == "user"
    assert sample["messages"][1]["role"] == "assistant"
    assert sample["messages"][1]["content"] == "nice to meet you"
    user_content = sample["messages"][0]["content"]
    image_entries = [c for c in user_content if c["type"] == "image"]
    text_entries = [c for c in user_content if c["type"] == "text"]
    assert len(image_entries) == 2
    assert len(text_entries) == 1
    assert text_entries[0]["text"] == "Translate this American Sign Language video into English text."


def test_load_manifest_dataset_applies_format(tmp_path):
    frame_paths = make_fake_frames(tmp_path, "clip_c", 4)
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        manifest,
        [
            {
                "clip_name": "clip_c",
                "sentence": "goodbye",
                "frame_paths": frame_paths,
            },
        ],
    )
    ds = load_manifest_dataset(manifest)
    sample = ds[0]
    assert "messages" in sample
    assert sample["messages"][1]["content"] == "goodbye"
