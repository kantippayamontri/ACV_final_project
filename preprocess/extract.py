"""Frame extraction and manifest building for How2Sign val set."""

from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

import cv2
import numpy as np

T = TypeVar("T")


def extract_frames(
    video_path: Path,
    out_dir: Path,
    n_frames: int = 8,
) -> list[Path]:
    """Extract `n_frames` evenly-spaced frames from `video_path` as JPEGs.

    Returns list of output JPEG paths (sorted by index).
    Raises ValueError if the video cannot be opened or has no frames.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Video has no readable frames: {video_path}")

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    paths: list[Path] = []
    for seq_idx, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        out_path = out_dir / f"frame_{seq_idx:02d}.jpg"
        cv2.imwrite(str(out_path), frame)
        paths.append(out_path)
    cap.release()
    return paths


def build_manifest(
    tsv_path: Path, # csv file that contain each clips information
    videos_dir: Path, # clips path
    out_dir: Path,
    n_frames: int = 8, #number of frame per video
    show_progress: bool = False,
    workers: int = 1,
) -> Path:
    """Extract frames for every clip in the TSV and write manifest.jsonl.

    Skips rows whose MP4 is not present in `videos_dir`.
    Returns the path to manifest.jsonl.
    """
    frames_root = out_dir / "frames"
    manifest_path = out_dir / "manifest.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    if workers < 1:
        raise ValueError("workers must be at least 1")

    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    # Load already-processed clip names so we can resume interrupted runs.
    done: set[str] = set()
    if manifest_path.exists():
        with open(manifest_path) as _f:
            for line in _f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["clip_name"])

    skipped = 0
    tasks: list[tuple[str, str, Path, Path, int]] = []
    for row in rows:
        clip_name = row["SENTENCE_NAME"].strip()
        if clip_name in done: #already process
            continue
        sentence = row["SENTENCE"].strip()
        mp4_path = videos_dir / f"{clip_name}.mp4"
        if not mp4_path.exists(): # check for video exist for extracting frames or not
            skipped += 1
            continue
        clip_out_dir = frames_root / clip_name
        tasks.append((clip_name, sentence, mp4_path, clip_out_dir, n_frames))

    written = 0
    with open(manifest_path, "a") as mf:
        results = _extract_records(tasks, workers=workers)
        for clip_name, record, error in _progress(
            results,
            enabled=show_progress,
            label="Extracting clips",
            total=len(tasks),
        ):
            if error is not None:
                print(f"[WARN] Skipping {clip_name}: {error}")
                skipped += 1
                continue
            mf.write(json.dumps(record) + "\n")
            written += 1

    print(f"Manifest updated: {manifest_path} (new={written}, skipped={skipped}, already_done={len(done)})")
    return manifest_path


def _progress(
    items: Iterable[T],
    *,
    enabled: bool,
    label: str,
    total: int | None = None,
) -> Iterator[T]:
    """Yield items while printing a compact terminal progress bar."""
    if not enabled:
        yield from items
        return

    if total is None:
        item_list = list(items)
        total = len(item_list)
        items = item_list

    if total == 0:
        print(f"{label}: no rows found", file=sys.stderr)
        return

    bar_width = 30

    def render(done: int) -> None:
        filled = int(bar_width * done / total)
        bar = "#" * filled + "-" * (bar_width - filled)
        percent = int(100 * done / total)
        print(
            f"\r{label}: |{bar}| {done}/{total} ({percent:3d}%)",
            end="",
            file=sys.stderr,
            flush=True,
        )

    render(0)
    for index, item in enumerate(items, start=1):
        yield item
        render(index)
    print(file=sys.stderr)


def _extract_records(
    tasks: list[tuple[str, str, Path, Path, int]],
    *,
    workers: int,
) -> Iterator[tuple[str, dict[str, object] | None, str | None]]:
    if workers == 1:
        for task in tasks:
            yield _extract_record(task)
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_extract_record, task) for task in tasks]
        for future in as_completed(futures):
            yield future.result()


def _extract_record(
    task: tuple[str, str, Path, Path, int],
) -> tuple[str, dict[str, object] | None, str | None]:
    clip_name, sentence, mp4_path, clip_out_dir, n_frames = task
    try:
        frame_paths = extract_frames(mp4_path, clip_out_dir, n_frames)
    except ValueError as e:
        return clip_name, None, str(e)

    record = {
        "clip_name": clip_name,
        "sentence": sentence,
        "frame_paths": [str(p) for p in frame_paths],
    }
    return clip_name, record, None
