# AGENTS.md

## Project

ASL video-to-text translation. Fine-tuning a VLM to translate American Sign Language (ASL) videos into English text.

- Base model: Qwen2.5-VL-2B (multimodal: images + text)
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

| Package | Version | Purpose |
|---|---|---|
| torch | 2.10.0+cu130 | PyTorch (CUDA 13.0) |
| torchvision | 0.25.0+cu130 | Vision ops |
| transformers | HF | LLMs / vision models |
| diffusers | HF | Diffusion models |
| peft | HF | LoRA / parameter-efficient fine-tuning |
| trl | HF | RLHF / SFT training |
| unsloth + unsloth_zoo | – | Fast fine-tuning optimization |
| accelerate | HF | Multi-GPU / mixed precision |
| bitsandbytes | – | 4-bit/8-bit quantization |
| xformers | – | Memory-efficient attention |
| datasets | HF | Dataset loading |
| numpy, pandas, PIL | – | Data utilities |

Do **not** reinstall these with pip; they are already present in `.venv`.

## No tests, lint, or CI configured yet

There are no test files, pytest config, linting rules, or CI workflows. `main.py` is a placeholder.

## CUDA

The torch build targets CUDA 13.0 (`cu130`). GPU ops require a compatible NVIDIA driver.
