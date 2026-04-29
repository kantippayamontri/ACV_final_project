"""Tests for preprocess.extract module.

Uses mocking to avoid slow video I/O operations.
"""
import csv
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    sys.modules["cv2"] = MagicMock()

from preprocess.extract import extract_frames, build_manifest


def make_fake_tsv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["VIDEO_ID", "VIDEO_NAME", "SENTENCE_ID", "SENTENCE_NAME", "START_REALIGNED", "END_REALIGNED", "SENTENCE"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


@patch("preprocess.extract.cv2.VideoCapture")
def test_extract_frames_produces_correct_count(MockVideoCapture, tmp_path):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 20  # 20 frames total
    mock_cap.read.return_value = (True, np.zeros((64, 64, 3), dtype=np.uint8))
    MockVideoCapture.return_value = mock_cap

    video_path = tmp_path / "clip_a.mp4"
    out_dir = tmp_path / "frames" / "clip_a"
    paths = extract_frames(video_path, out_dir, n_frames=8)

    assert len(paths) == 8
    for p in paths:
        assert p.suffix == ".jpg"


@patch("preprocess.extract.cv2.VideoCapture")
def test_extract_frames_handles_missing_video(MockVideoCapture, tmp_path):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    MockVideoCapture.return_value = mock_cap

    video_path = tmp_path / "missing.mp4"
    out_dir = tmp_path / "frames" / "missing"

    with pytest.raises(ValueError, match="Cannot open video"):
        extract_frames(video_path, out_dir, n_frames=8)


def test_build_manifest_skips_missing_video(tmp_path):
    tsv_path = tmp_path / "val.csv"
    videos_dir = tmp_path / "raw_videos"
    videos_dir.mkdir()

    make_fake_tsv(
        tsv_path,
        [
            {"VIDEO_ID": "v1", "VIDEO_NAME": "n1", "SENTENCE_ID": "s1", "SENTENCE_NAME": "clip_present", "START_REALIGNED": "0", "END_REALIGNED": "1", "SENTENCE": "hello world"},
            {"VIDEO_ID": "v2", "VIDEO_NAME": "n2", "SENTENCE_ID": "s2", "SENTENCE_NAME": "clip_missing", "START_REALIGNED": "0", "END_REALIGNED": "1", "SENTENCE": "should be skipped"},
        ],
    )
    # Create a dummy file (not a real video, but exists)
    (videos_dir / "clip_present.mp4").write_bytes(b"fake")

    out_dir = tmp_path / "processed"

    with patch("preprocess.extract.extract_frames") as mock_extract:
        mock_extract.return_value = [tmp_path / "fake.jpg"] * 4
        manifest_path = build_manifest(tsv_path, videos_dir, out_dir, n_frames=4)

    lines = manifest_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["clip_name"] == "clip_present"
    assert record["sentence"] == "hello world"
    assert len(record["frame_paths"]) == 4


def test_build_manifest_jsonl_format(tmp_path):
    tsv_path = tmp_path / "val.csv"
    videos_dir = tmp_path / "raw_videos"
    videos_dir.mkdir()

    make_fake_tsv(
        tsv_path,
        [
            {"VIDEO_ID": "v1", "VIDEO_NAME": "n1", "SENTENCE_ID": "s1", "SENTENCE_NAME": "myclip", "START_REALIGNED": "0", "END_REALIGNED": "1", "SENTENCE": "sign this"},
        ],
    )
    (videos_dir / "myclip.mp4").write_bytes(b"fake")

    out_dir = tmp_path / "processed"

    with patch("preprocess.extract.extract_frames") as mock_extract:
        mock_extract.return_value = [tmp_path / "fake.jpg"] * 4
        manifest_path = build_manifest(tsv_path, videos_dir, out_dir, n_frames=4)

    record = json.loads(manifest_path.read_text().splitlines()[0])
    assert set(record.keys()) == {"clip_name", "sentence", "frame_paths"}
    assert all(Path(p).name.endswith(".jpg") for p in record["frame_paths"])


def test_build_manifest_rejects_invalid_worker_count(tmp_path):
    tsv_path = tmp_path / "val.csv"
    videos_dir = tmp_path / "raw_videos"
    videos_dir.mkdir()
    make_fake_tsv(tsv_path, [])

    with pytest.raises(ValueError, match="workers must be at least 1"):
        build_manifest(tsv_path, videos_dir, tmp_path / "processed", workers=0)
