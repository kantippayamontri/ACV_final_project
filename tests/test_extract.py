import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from preprocess.extract import extract_frames, build_manifest


def make_fake_video(path: Path, num_frames: int = 20) -> None:
    """Write a tiny synthetic MP4 with solid-colour frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 10, (64, 64))
    for i in range(num_frames):
        frame = np.full((64, 64, 3), i * 10, dtype=np.uint8)
        out.write(frame)
    out.release()


def make_fake_tsv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["VIDEO_ID", "VIDEO_NAME", "SENTENCE_ID", "SENTENCE_NAME", "START_REALIGNED", "END_REALIGNED", "SENTENCE"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_extract_frames_produces_correct_count(tmp_path):
    video_path = tmp_path / "clip_a.mp4"
    make_fake_video(video_path, num_frames=20)
    out_dir = tmp_path / "frames" / "clip_a"
    paths = extract_frames(video_path, out_dir, n_frames=8)
    assert len(paths) == 8
    for p in paths:
        assert p.exists()
        assert p.suffix == ".jpg"


def test_extract_frames_names_sequentially(tmp_path):
    video_path = tmp_path / "clip_b.mp4"
    make_fake_video(video_path)
    out_dir = tmp_path / "frames" / "clip_b"
    paths = extract_frames(video_path, out_dir, n_frames=4)
    names = [p.name for p in paths]
    assert names == ["frame_00.jpg", "frame_01.jpg", "frame_02.jpg", "frame_03.jpg"]


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
    make_fake_video(videos_dir / "clip_present.mp4")
    out_dir = tmp_path / "processed"
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
    make_fake_video(videos_dir / "myclip.mp4")
    out_dir = tmp_path / "processed"
    manifest_path = build_manifest(tsv_path, videos_dir, out_dir, n_frames=4)
    record = json.loads(manifest_path.read_text().splitlines()[0])
    assert set(record.keys()) == {"clip_name", "sentence", "frame_paths"}
    assert all(Path(p).name.endswith(".jpg") for p in record["frame_paths"])
