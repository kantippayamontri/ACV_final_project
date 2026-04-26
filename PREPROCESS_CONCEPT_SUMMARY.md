# Preprocessing Concept Summary

## 1. Overview — What Preprocessing Does and Why

The fine-tuning model (Qwen3-VL) cannot read raw `.mp4` video files directly. It is a vision-language model that processes **still images** and **text** — not video streams.

Preprocessing converts the raw How2Sign dataset into a format the model can consume:

```
Raw input:   .mp4 clip files  +  TSV annotation file (clip name → English sentence)
                    ↓
Preprocessing:  extract frames  +  build manifest
                    ↓
Output:      JPEG frames on disk  +  manifest.jsonl (links frames to translation)
```

The manifest is then read by the training pipeline, which formats each record into the ChatML conversation structure the model expects.

---

## 2. Input: The How2Sign Dataset Structure

### TSV annotation file

`datasets/raw/val_rgb_front_clips/how2sign_realigned_val.csv`

A tab-separated file with one row per clip. Key columns:

| Column | Description |
|---|---|
| `SENTENCE_NAME` | Unique clip identifier (used as the filename stem) |
| `SENTENCE` | The English translation of the signed sentence |
| `START_REALIGNED` | Start time of the sign in the source video (seconds) |
| `END_REALIGNED` | End time of the sign in the source video (seconds) |

Example row:
```
VIDEO_ID    VIDEO_NAME    SENTENCE_ID    SENTENCE_NAME                  START_REALIGNED    END_REALIGNED    SENTENCE
...         ...           ...            -d5dN54tH2E_0-1-rgb_front      0.0                4.2              We're going to work on a arm drill...
```

### Video clip files

`datasets/raw/val_rgb_front_clips/raw_videos/`

Each clip is a `.mp4` file named `<SENTENCE_NAME>.mp4`. These are short clips (typically 2–10 seconds) of a single signed sentence filmed from the front.

The dataset has **1,741 clips** in the validation split.

---

## 3. Step 1: Frame Extraction (`extract_frames()`)

### Why frames instead of video?

Vision-language models process a fixed set of still images per forward pass. There is no built-in video stream support — the temporal information must be encoded as a sequence of representative frames.

### How frames are selected: evenly spaced sampling

Rather than taking consecutive frames (which would all look nearly identical), `extract_frames()` uses `numpy.linspace` to pick `n_frames` indices **evenly spread across the full video duration**:

```python
indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
```

For example, a 100-frame video with `n_frames=8` picks frames at indices:
```
[0, 14, 28, 42, 57, 71, 85, 99]
```

This samples the **start, middle, and end** of each sign — capturing the full arc of the hand movement, not just a snapshot of a single moment.

### Why evenly spaced and not random?

- ASL signs have a clear temporal structure: preparation → stroke → hold → retraction
- Random sampling risks clustering frames in one part of the sign
- Evenly spaced sampling is deterministic and reproducible (same frames every run)
- At `n_frames=8`, each frame covers roughly 1/8 of the total sign duration

### What happens if a frame can't be read?

If OpenCV fails to decode a specific frame (corrupted data), a **black 224×224 placeholder image** is substituted:

```python
if not ret:
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
```

This ensures the clip is not silently dropped from the dataset. The number of frames per clip always equals `n_frames`.

### Output structure

```
datasets/processed/frames/
└── -d5dN54tH2E_0-1-rgb_front/
    ├── frame_00.jpg   ← frame at index 0   (start of sign)
    ├── frame_01.jpg   ← frame at index 14
    ├── frame_02.jpg   ← frame at index 28
    ├── frame_03.jpg   ← frame at index 42
    ├── frame_04.jpg   ← frame at index 57
    ├── frame_05.jpg   ← frame at index 71
    ├── frame_06.jpg   ← frame at index 85
    └── frame_07.jpg   ← frame at index 99  (end of sign)
```

Frames are saved as **JPEG** at the original resolution (1280×720). JPEG is used over PNG because:
- ~3–5× smaller file size for natural images
- Quality is sufficient for 1280×720 RGB frames
- Faster to open during training (less I/O)

---

## 4. Step 2: Manifest Building (`build_manifest()`)

### What is a manifest?

A **manifest** is a JSONL file (one JSON object per line) where each line represents one complete training sample. It links a clip's extracted frames to its English translation.

Example record from `datasets/processed/manifest.jsonl`:

```json
{
  "clip_name": "-d5dN54tH2E_0-1-rgb_front",
  "sentence": "We're going to work on a arm drill that will help you have graceful hand movements in front of you.",
  "frame_paths": [
    "datasets/processed/frames/-d5dN54tH2E_0-1-rgb_front/frame_00.jpg",
    "datasets/processed/frames/-d5dN54tH2E_0-1-rgb_front/frame_01.jpg",
    "datasets/processed/frames/-d5dN54tH2E_0-1-rgb_front/frame_02.jpg",
    "datasets/processed/frames/-d5dN54tH2E_0-1-rgb_front/frame_03.jpg",
    "datasets/processed/frames/-d5dN54tH2E_0-1-rgb_front/frame_04.jpg",
    "datasets/processed/frames/-d5dN54tH2E_0-1-rgb_front/frame_05.jpg",
    "datasets/processed/frames/-d5dN54tH2E_0-1-rgb_front/frame_06.jpg",
    "datasets/processed/frames/-d5dN54tH2E_0-1-rgb_front/frame_07.jpg"
  ]
}
```

Each field:

| Field | Description |
|---|---|
| `clip_name` | The `SENTENCE_NAME` from the TSV — used as the directory name for frames |
| `sentence` | The English translation — this is the **training label** (what the model learns to output) |
| `frame_paths` | Ordered list of absolute/relative paths to the extracted JPEG frames |

### Resume support

Preprocessing can take a long time (1,741 clips × frame extraction). If interrupted, it is safe to restart — `build_manifest()` reads the existing `manifest.jsonl` at the start and collects all already-processed clip names:

```python
done: set[str] = set()
if manifest_path.exists():
    for line in open(manifest_path):
        done.add(json.loads(line)["clip_name"])
```

Any clip already in `done` is skipped. The manifest is opened in **append mode** (`"a"`), so previously written records are preserved.

### Skipped clips

Two cases cause a clip to be skipped (counted in the `skipped` log):
1. The `.mp4` file does not exist in `raw_videos/` — clip is in the TSV but the video file was not downloaded
2. OpenCV raises a `ValueError` (cannot open the video or video has no frames)

Neither case crashes the pipeline — it logs a warning and moves on.

---

## 5. Step 3: ChatML Formatting (`_record_to_sample()`)

### What is ChatML?

ChatML is the **conversation format** that Qwen3-VL (and most instruction-tuned LLMs) expect as input. It structures a sample as a list of turns, each with a `role` (`user` or `assistant`) and `content`.

For vision-language models, the `user` turn can contain a mix of images and text.

### How a manifest record maps to ChatML

```python
{
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "datasets/processed/frames/.../frame_00.jpg"},
                {"type": "image", "image": "datasets/processed/frames/.../frame_01.jpg"},
                # ... 6 more frames ...
                {"type": "text", "text": "Translate this American Sign Language video into English text."}
            ]
        },
        {
            "role": "assistant",
            "content": "We're going to work on a arm drill..."
        }
    ]
}
```

- The 8 frames come first in the user content, in temporal order
- The prompt text comes last in the user content
- The assistant content is the ground truth English sentence — this is what the model is trained to predict

### Why frame paths are stored as strings, not PIL Images

The manifest stores file paths as strings. PIL Images are only opened **at data collation time** (inside `UnslothVisionDataCollator`), not when the dataset is loaded.

This is deliberate: loading all 1,741 clips × 8 frames as PIL Images upfront would require holding thousands of 1280×720 images in RAM simultaneously — easily exceeding available memory. Lazy loading keeps RAM usage flat.

---

## 6. Full Pipeline End-to-End

```
datasets/raw/
├── how2sign_realigned_val.csv     ← 1,741 rows: clip_name + sentence
└── raw_videos/
    ├── -d5dN54tH2E_0-1-rgb_front.mp4
    ├── ...
    └── <clip_name>.mp4
           │
           │  build_manifest()
           │    reads TSV row by row
           │    for each clip → extract_frames()
           ▼
datasets/processed/
├── manifest.jsonl                 ← 1 JSON line per clip
└── frames/
    └── <clip_name>/
        ├── frame_00.jpg
        └── ... frame_07.jpg
           │
           │  _record_to_sample()  (called in train.py)
           │    converts manifest record → ChatML dict
           ▼
[
  {
    "messages": [
      {"role": "user", "content": [8 image paths + prompt text]},
      {"role": "assistant", "content": "English sentence"}
    ]
  },
  ...
]
           │
           │  SFTTrainer + UnslothVisionDataCollator
           │    opens images lazily, tokenizes, trains
           ▼
        LoRA adapters saved to runs/run_<timestamp>_steps<N>/
```

---

## 7. How to Run Preprocessing

```bash
# Default paths (How2Sign val split)
python preprocess_data.py

# Custom paths
python preprocess_data.py \
  --tsv datasets/raw/val_rgb_front_clips/how2sign_realigned_val.csv \
  --videos datasets/raw/val_rgb_front_clips/raw_videos \
  --out datasets/processed \
  --n-frames 8

# Extract more frames per clip (more temporal detail, more tokens)
python preprocess_data.py --n-frames 16

# Via main.py wrapper
python main.py --preprocess
```

Output after a complete run:
```
Manifest updated: datasets/processed/manifest.jsonl (new=1741, skipped=0, already_done=0)
```

---

## 8. Key Design Decisions

### Why n_frames=8?

| n_frames | Tokens per sample (after MAX_PIXELS cap) | Notes |
|---|---|---|
| 4 | ~1,920 | Fast, fits easily, but misses fine-grained temporal detail |
| 8 | ~3,840 | Default — good balance of coverage vs. token budget |
| 16 | ~7,680 | Exceeds MAX_SEQ_LENGTH=4096 — would require larger cap or fewer pixels |

8 frames was chosen to fit within `MAX_SEQ_LENGTH=4096` while giving enough temporal coverage for a sign (start / middle / end well represented).

### Why JPEG and not PNG?

JPEG is lossy but the compression artefacts at 1280×720 are imperceptible for the purposes of training a vision model on hand/body pose. PNG would use 3–5× more disk space with no meaningful quality benefit.

### Why preprocessing is model-agnostic

Preprocessing saves frames at their **original 1280×720 resolution** — no resizing is done here. The decision of how many pixels to keep per frame (`MAX_PIXELS`) lives in `train.py` and `inference.py`, because:

- Different experiments may want different resolutions
- Changing `MAX_PIXELS` requires no re-extraction — just a constant change
- If frames were resized during preprocessing, any resolution experiment would require re-running the full extraction pipeline from the original `.mp4` files

### Why resumable?

Extracting 1,741 clips can take 10–30 minutes depending on disk speed. The resume mechanism (checking `done` set against existing manifest lines) means an interrupted run picks up exactly where it left off with no duplicate records and no re-work.
