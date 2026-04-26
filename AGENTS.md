# AGENTS.md

## Project

ASL video-to-text translation. Fine-tuning a VLM to translate American Sign Language (ASL) videos into English text.

- Base model: Qwen3-VL-2B (multimodal: images + text)
- Training: Unsloth + QLoRA (4-bit quantization) — fits on consumer GPU
- Dataset: [How2Sign](https://how2sign.github.io/)
- Input format: videos sampled to 8–16 frames, formatted as ChatML with vision tokens (`<|vision_start|>`)
- Key libraries: OpenCV, PyTorch, Unsloth, Transformers
- OS: Linux/Ubuntu 24.04

Python 3.13 ML/CV project (`acv-final-project`). Uses `uv` for environment management (venv at `.venv/`, created via uv 0.11.1).

## Environment

- Python: 3.13 (pinned in `.python-version`)
- Package manager: `uv` (not pip directly)
- Activate venv: `source .venv/bin/activate`
- Add dependencies: `uv add <package>` (updates `pyproject.toml`; no lockfile present)
- Run script: `uv run python main.py` or activate venv first

## Key installed packages (already in `.venv`)

| Package               | Version      | Purpose                                |
| --------------------- | ------------ | -------------------------------------- |
| torch                 | 2.10.0+cu130 | PyTorch (CUDA 13.0)                    |
| torchvision           | 0.25.0+cu130 | Vision ops                             |
| transformers          | HF           | LLMs / vision models                   |
| diffusers             | HF           | Diffusion models                       |
| peft                  | HF           | LoRA / parameter-efficient fine-tuning |
| trl                   | HF           | RLHF / SFT training                    |
| unsloth + unsloth_zoo | –            | Fast fine-tuning optimization          |
| accelerate            | HF           | Multi-GPU / mixed precision            |
| bitsandbytes          | –            | 4-bit/8-bit quantization               |
| xformers              | –            | Memory-efficient attention             |
| datasets              | HF           | Dataset loading                        |
| numpy, pandas, PIL    | –            | Data utilities                         |

Do **not** reinstall these with pip; they are already present in `.venv`.

## Tests

Tests live in `tests/` and use `pytest` with mocking — no GPU or real video I/O required:

| Test file                       | Covers                                                 |
| ------------------------------- | ------------------------------------------------------ |
| `tests/test_extract.py`         | `extract_frames()`, `build_manifest()` — mocked OpenCV |
| `tests/test_dataset.py`         | `load_manifest_dataset()`, `format_sample()`           |
| `tests/test_train.py`           | `load_training_dataset()` — ChatML format, image paths |
| `tests/test_train_cli.py`       | `cli()` argument parsing                               |
| `tests/test_preprocess_data.py` | `main()` argument parsing                              |

Run tests: `uv run pytest tests/ -v`

## Project structure

```
main.py                  # Thin CLI wrapper: --preprocess | --train
preprocess_data.py       # Standalone preprocessing entry point
train.py                 # Core training: loads model, applies LoRA, runs SFT
inference.py             # Loads fine-tuned LoRA adapters and evaluates with BLEU
preprocess/
    __init__.py
    extract.py           # OpenCV frame extraction + manifest builder
    dataset.py           # Load manifest as HF Dataset (alternative loader)
tests/
    __init__.py
    test_extract.py
    test_dataset.py
    test_train.py
    test_train_cli.py
    test_preprocess_data.py
datasets/
    raw/                 # Raw How2Sign videos + TSV
    processed/
        manifest.jsonl   # {clip_name, sentence, frame_paths}
        frames/          # Extracted JPEG frames per clip
asl_lora_output/         # Trained LoRA adapters (870 steps)
    checkpoint-500/
    checkpoint-870/
```

## Training details

- LoRA: r=16, alpha=16, dropout=0, target_modules="all-linear"
- Both vision and language layers fine-tuned
- Batch size: 1 (effective 4 with gradient accumulation)
- Learning rate: 2e-4, cosine schedule, adamw_8bit
- Max sequence length: 2048
- Already trained: 870 steps completed

## Inference

`inference.py` loads fine-tuned LoRA adapters and evaluates translation quality using BLEU-4 and exact match:

```bash
# Evaluate on first 10 samples using checkpoint-870
uv run inference.py --run-dir asl_lora_output/checkpoint-870 --max-samples 10

# Evaluate on full manifest using a specific run
uv run inference.py --run-dir runs/run_20260426_143022_steps300 --manifest datasets/processed/manifest.jsonl
```

- Loads base model + LoRA weights via `FastVisionModel.from_pretrained(run_dir)`
- Uses greedy decoding (`do_sample=False`) with `max_new_tokens=128`
- Reports BLEU-4 score and exact match percentage
- Requires `sacrebleu` package
