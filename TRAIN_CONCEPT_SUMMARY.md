# Training Concept Summary

## Table of Contents

1. [MAX_SEQ_LENGTH and MAX_PIXELS — the token budget problem](#1-max_seq_length-and-max_pixels--the-token-budget-problem)
2. [Validation dataset and early stopping](#2-validation-dataset-and-early-stopping)
3. [Epoch-based step calculation](#3-epoch-based-step-calculation)
4. [Training hyperparameters](#4-training-hyperparameters)
5. [Summary of all changes](#5-summary-of-all-changes)

---

## 1. MAX_SEQ_LENGTH and MAX_PIXELS — the token budget problem

### What is MAX_SEQ_LENGTH?

`MAX_SEQ_LENGTH` is the maximum number of tokens the model can process in a single forward pass. Everything in one training sample — images + prompt + answer — must fit within this limit. If it exceeds the limit, the trainer **silently truncates** the sequence from the end, meaning the model never sees the cut-off content.

### How Qwen3-VL converts images to tokens

Qwen3-VL's vision encoder works by dividing each image into a grid of **32×32 pixel patches**. Each patch becomes one vision token.

> Note: Qwen2-VL used 28×28 patches. Qwen3-VL upgraded to 32×32. Using 28 with Qwen3-VL produces misaligned token counts.

```
tokens per image = (width / 32) × (height / 32)
```

For example, a 256×256 image:
```
(256 / 32) × (256 / 32) = 8 × 8 = 64 tokens
```

### Your frames: 1280×720

The extracted frames from How2Sign are saved at **1280×720 pixels**. Plugging into the formula:

```
(1280 / 32) × (720 / 32) = 40 × 22 = 880 tokens per frame
```

With 8 frames per clip, the total token budget for images alone:

```
8 frames × 880 tokens = 7,040 visual tokens
```

### Why MAX_SEQ_LENGTH=2048 failed

A full training sample contains:

| Component | Tokens |
|---|---|
| 8 frames (1280×720, no resize) | ~7,040 |
| Chat template overhead | ~30 |
| Prompt text ("Translate this ASL video...") | ~15 |
| Ground truth sentence (assistant answer) | ~10–30 |
| **Total needed** | **~7,095** |

At `MAX_SEQ_LENGTH=2048`, the trainer truncates everything beyond token 2048. Since image tokens come first in the sequence, only the first ~2.3 frames worth of vision tokens survive:

```
2048 tokens / 880 tokens per frame ≈ 2.3 frames
```

The model was effectively training on ~2 frames out of 8 — seeing only a small portion of each sign — which severely limits what it can learn about ASL.

### What exactly gets cut off

The sequence is built in this order before truncation:

```
[chat template] → [frame_1 tokens] → [frame_2 tokens] → ... → [frame_8 tokens] → [prompt text] → [answer]
```

With 880 tokens per frame at 1280×720:

```
frame_1 : tokens    1 –   880   ✓ fully seen
frame_2 : tokens  881 – 1,760   ✓ fully seen
frame_3 : tokens 1,761 – 2,640  ✗ cut at token 2,048 — only ~33% survives (partial frame)
frame_4 : tokens 2,641 – 3,520  ✗ never seen
frame_5 : tokens 3,521 – 4,400  ✗ never seen
frame_6 : tokens 4,401 – 5,280  ✗ never seen
frame_7 : tokens 5,281 – 6,160  ✗ never seen
frame_8 : tokens 6,161 – 7,040  ✗ never seen
prompt  : tokens 7,041 – 7,056  ✗ never seen
answer  : tokens 7,057 – 7,090  ✗ never seen  ← ground truth lost
```

The most critical consequence is that **the ground truth answer is always cut off**. The trainer computes the loss on the assistant's response — if it never appears in the sequence, the model has nothing to learn from. It is essentially training on noise.

So the real effect of `MAX_SEQ_LENGTH=2048` is not "only reads 1 frame" but rather:
- Sees frames 1–2 fully
- Sees ~33% of frame 3 (cut mid-patch)
- Never sees frames 4–8
- Never sees the prompt
- Never sees the answer it is supposed to predict

---

### The Fix: Two Changes Working Together

#### Change 1: Introduce MAX_PIXELS to control image resize

`MAX_PIXELS` sets a ceiling on how many pixels an image can have before the processor resizes it down. If an image exceeds this limit, the processor scales it down proportionally (preserving aspect ratio) and then snaps each dimension to the nearest multiple of 32 (Qwen3-VL's patch size).

```python
MAX_PIXELS = 512 * 32 * 32  # = 524,288 pixels
```

The number `512` represents the maximum number of 32×32 patches (tokens) allowed per frame.

#### How the resize calculation works

Given MAX_PIXELS = 524,288 and input frame 1280×720 = 921,600 pixels:

**Step 1: Compute scale factor**
```
scale = sqrt(MAX_PIXELS / input_pixels)
      = sqrt(524,288 / 921,600)
      = sqrt(0.5690)
      = 0.7543
```
`sqrt` is used because both width and height are scaled by the same factor, so total pixels scale by `factor²`. To hit a target pixel count, solve for the factor.

**Step 2: Scale each dimension**
```
new_width  = 1280 × 0.7543 = 965.5
new_height = 720  × 0.7543 = 543.1
```

**Step 3: Round down to nearest multiple of 32**
```
new_width  = floor(965.5 / 32) × 32 = 30 × 32 = 960
new_height = floor(543.1 / 32) × 32 = 16 × 32 = 512
```

**Result: 1280×720 → 960×512**

**Step 4: Token count after resize**
```
(960 / 32) × (512 / 32) = 30 × 16 = 480 tokens per frame
8 frames × 480 = 3,840 visual tokens
```

This is where `MAX_PIXELS = 512 × 32 × 32` comes from: we budget **at most 512 tokens per frame**, which after aspect-ratio rounding gives us 480 in practice.

#### Change 2: Raise MAX_SEQ_LENGTH to 5120

With `MAX_PIXELS` capping each frame to ~480 tokens, the full sample now needs:

| Component | Tokens |
|---|---|
| 8 frames (960×512, after resize) | ~3,840 |
| Chat template overhead | ~30 |
| Prompt text | ~15 |
| Ground truth sentence | ~10–30 |
| **Total needed** | **~3,895–3,915** |

Setting `MAX_SEQ_LENGTH=5120` gives ~1,205 tokens of headroom — no truncation occurs.

```python
MAX_SEQ_LENGTH = 5120
```

### Why Not Just Raise MAX_SEQ_LENGTH Without MAX_PIXELS?

Without the pixel cap, 8 frames at 1280×720 need ~7,040 tokens. To fit that:
- `MAX_SEQ_LENGTH` would need to be at least 7,200
- Processing 7,200 tokens per step requires significantly more VRAM (KV cache grows with sequence length)
- On an RTX 2060 with ~6GB VRAM, this would cause an out-of-memory (OOM) error

The two changes work together:
- `MAX_PIXELS` reduces visual tokens per frame (3,840 instead of 7,040)
- `MAX_SEQ_LENGTH=5120` provides enough room for the reduced token count with ample headroom

### Why Not Resize Frames During Preprocessing?

Resizing could happen at two points:
1. **During preprocessing** (`extract_frames()`) — save smaller JPEGs to disk
2. **During training/inference** (processor) — resize in memory each step ← chosen approach

Approach 2 was chosen because:
- Raw 1280×720 frames are preserved on disk permanently
- `MAX_PIXELS` can be changed without re-running preprocessing (which requires the original videos)
- If you want to experiment with more tokens per frame (e.g. `MAX_PIXELS = 768×32×32`), just change one constant — no re-extraction needed
- Preprocessing should be model-agnostic; resolution decisions belong to the model pipeline

### Processor pixel cap — transformers 5.x compatibility

In transformers ≥5.0, `Qwen2VLImageProcessorFast` stores the pixel limits inside a `SizeDict`:

```
self.size["longest_edge"]  ← max_pixels
self.size["shortest_edge"] ← min_pixels
```

Two approaches that **do not work**:
- `tokenizer.image_processor.max_pixels = X` → `AttributeError`: property has no setter
- Passing `min_pixels=X, max_pixels=X` to `FastVisionModel.from_pretrained()` → `TypeError`: model `__init__` rejects unknown kwargs

The correct approach — direct dict mutation after loading:

```python
tokenizer.image_processor.size["longest_edge"] = MAX_PIXELS
tokenizer.image_processor.size["shortest_edge"] = 4 * 32 * 32
```

---

## 2. Validation dataset and early stopping

### What is early stopping?

Early stopping monitors the model's performance on a held-out **validation set** during training. If the validation loss stops improving, training halts automatically — preventing the model from overfitting to the training data.

Without early stopping, you must choose `max_steps` in advance and hope the model hasn't already started overfitting by then.

### How it is implemented

When `--val-manifest` is provided, `train()`:

1. Loads a second dataset from the val manifest (same format as train).
2. Passes `eval_dataset=val_dataset` to `SFTTrainer`.
3. Adds `EarlyStoppingCallback(early_stopping_patience=3)`.
4. Configures `SFTConfig` with matching eval and save strategies.

```python
# eval and save must use the same strategy for load_best_model_at_end to work
eval_strategy  = "steps"
eval_steps     = 500          # evaluate every 500 training steps
save_strategy  = "steps"
save_steps     = 500          # save checkpoint every 500 steps (must match eval_steps)
load_best_model_at_end = True
metric_for_best_model  = "eval_loss"
greater_is_better      = False  # lower loss = better
```

The `EarlyStoppingCallback` with `patience=3` means:
- After each eval, if `eval_loss` is not a new minimum → increment patience counter
- If counter reaches 3 (i.e. 3 consecutive evals with no improvement = 1,500 steps of no improvement) → stop training
- The best checkpoint (lowest `eval_loss`) is restored at the end

### When no val set is provided

Behaviour is identical to before: no eval, `save_strategy="steps"` with `save_steps=500`. `max_steps` is the only stopping criterion.

### Why use the same manifest for both train and val (testing only)?

During development you can pass the same manifest as both `--manifest-path` and `--val-manifest`. This does not help the model learn (it will overfit) but it verifies the pipeline runs end-to-end without errors. In real training, use a proper held-out val split.

---

## 3. Epoch-based step calculation

### Why compute steps from epochs?

Previously `max_steps=15000` was hardcoded — a rough estimate assuming ~30k training samples. If your manifest has a different number of samples, the hardcoded value silently gives you fewer or more than the intended 2 epochs.

Computing steps from the manifest size makes training length predictable regardless of dataset size.

### The formula

```
effective_batch_size = per_device_batch_size × gradient_accumulation_steps
                     = 1 × 4
                     = 4

steps_per_epoch = ceil(num_samples / effective_batch_size)
               = ceil(num_samples / 4)

max_steps = steps_per_epoch × num_epochs
```

**Why `ceil`?** The last batch of an epoch may have fewer than 4 samples. `ceil` ensures that partial batch is still counted as a step, so the model sees every sample.

**Why `effective_batch_size = 4`?** With `per_device_train_batch_size=1` and `gradient_accumulation_steps=4`, the optimizer updates once every 4 forward passes. From the dataset's perspective, one optimizer step consumes 4 samples.

### Example

```
num_samples     = 31,128   (How2Sign train split)
steps_per_epoch = ceil(31,128 / 4) = 7,782
max_steps (2 epochs) = 7,782 × 2 = 15,564
```

At startup, `train()` prints:
```
Training: 31128 samples, 2 epoch(s) → 7782 steps/epoch = 15564 computed steps (max_steps=15564)
```

### Overriding with --max-steps

`--max-steps` overrides the epoch calculation entirely. This is useful for:
- Quick smoke tests: `--max-steps 50`
- Resuming from a checkpoint partway through

When `--max-steps` is used, `--epochs` still controls the output directory name unless `--output-dir` is explicitly set.

### Output directory naming

| Command | Output dir suffix |
|---|---|
| `--epochs 2` (default) | `_ep2` |
| `--epochs 3` | `_ep3` |
| `--max-steps 300` | `_steps300` |
| `--epochs 2 --max-steps 300` | `_steps300` (max-steps wins) |

---

## 4. Training hyperparameters

### Loss function

**Cross-entropy on assistant response tokens only.**

`SFTTrainer` performs Supervised Fine-Tuning (SFT). The loss is:

```
L = -∑ log P(token_t | token_1, ..., token_{t-1})
```

summed only over the **assistant turn tokens** (the ground-truth English sentence), not over the user turn (images + prompt). This is called "train on responses only" — the model is penalised only for getting the translation wrong, not for failing to predict the input frames or the prompt.

`UnslothVisionDataCollator` handles the label masking: it sets labels for all non-assistant tokens to `-100`, which PyTorch's `CrossEntropyLoss` ignores. Only the translation tokens contribute to the gradient.

What is NOT used:
- No BLEU loss (BLEU is non-differentiable; it is only used at evaluation time in `inference.py`)
- No contrastive loss
- No RLHF / reward model — this is pure SFT

---

### Optimizer

**`adamw_8bit`** (8-bit AdamW via `bitsandbytes`)

AdamW = Adam + weight decay. Adam maintains two moving averages per parameter:
- **m** (first moment) — exponential moving average of gradients
- **v** (second moment) — exponential moving average of squared gradients

Update rule:
```
m  = β1 * m + (1 - β1) * grad        # β1 = 0.9
v  = β2 * v + (1 - β2) * grad²       # β2 = 0.999
θ  = θ - lr * (m / (√v + ε)) - lr * λ * θ   # λ = weight_decay
```

The `_8bit` suffix means the optimizer states (m and v) are stored in 8-bit instead of 32-bit — reduces optimizer memory by ~4×. Critical on the RTX 2060 with 5.6 GB VRAM.

**`weight_decay=0.01`** — the λ term above. Adds a small penalty for large weights, acting as regularisation.

---

### Learning rate schedule

**`lr_scheduler_type="cosine"`** with **`learning_rate=2e-4`** and **`warmup_steps=5`**

```
Phase 1 (steps 0–5):    linear warmup   0     → 2e-4
Phase 2 (steps 5–end):  cosine decay    2e-4  → ~0
```

The cosine curve decays smoothly — large updates early, tiny updates near the end. Warmup prevents large gradient updates at the start when the LoRA weights are randomly initialised.

---

### Batch size and gradient accumulation

```
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
→ effective batch size       = 4
```

With batch size 1, each sample is one forward pass. Gradients are accumulated for 4 passes before a single optimizer step. This simulates batch size 4 without loading 4 samples into GPU memory simultaneously — necessary given VRAM constraints.

---

### Precision

**`fp16=True`** on the RTX 2060 (no bfloat16 support at compute capability 7.5).

| What | Precision |
|---|---|
| Base model weights | 4-bit (QLoRA) |
| LoRA adapter weights | 16-bit (updated during training) |
| Forward pass / gradients | fp16 |
| Optimizer states (m, v) | 8-bit (via `adamw_8bit`) |

Only the LoRA adapter weights are updated — the base model stays frozen at 4-bit throughout.

---

### LoRA hyperparameters

| Parameter | Value | Meaning |
|---|---|---|
| `r=16` | rank | Size of the low-rank decomposition. Higher = more capacity, more memory |
| `lora_alpha=16` | scaling | Effective scale = `alpha / r = 1.0` — updates applied at full scale |
| `lora_dropout=0` | dropout | No dropout on LoRA layers |
| `target_modules="all-linear"` | scope | LoRA applied to every linear layer (vision encoder + language model) |
| `bias="none"` | bias | Base model biases frozen; no LoRA bias terms added |

The effective LoRA scaling factor `alpha/r = 16/16 = 1.0` means LoRA updates are applied without amplification or dampening.

Why both `finetune_vision_layers=True` and `finetune_language_layers=True`: ASL translation requires both learning new visual features from signing (vision) and mapping them to English text (language). Freezing either side would bottleneck learning.

---

### Full parameter reference

| Parameter | Value | Why |
|---|---|---|
| Optimizer | `adamw_8bit` | Standard for LLM fine-tuning; 8-bit saves VRAM |
| Weight decay | `0.01` | Light regularisation |
| Learning rate | `2e-4` | Standard for LoRA fine-tuning |
| LR schedule | cosine | Smooth decay; avoids sudden loss spikes at end |
| Warmup steps | `5` | Stabilises early training; small because LoRA weights are small |
| Batch size | `1 × 4 accum = 4` | VRAM constraint |
| Precision | `fp16` | RTX 2060 lacks bf16 |
| Base model quant | 4-bit (QLoRA) | Fits 2B model in ~1.5 GB VRAM |
| LoRA rank | `16` | Good balance of capacity vs. memory |
| LoRA alpha | `16` | Scale factor = 1.0 |
| Seed | `3407` | Reproducibility |
| Max seq length | `4096` | Fits 8 capped frames + text |
| Max pixels | `512×28×28` | Caps 1280×720 → 840×448 → 480 tokens/frame |

---

## 5. Summary of all changes

| What | Before | After | Why |
|---|---|---|---|
| `MAX_SEQ_LENGTH` | `2048` | `5120` | 2048 truncated all but ~2.3 frames; 5120 fits 8 capped frames with 1,205 tokens headroom |
| `MAX_PIXELS` | not set | `512 × 32 × 32 = 524,288` | Qwen3-VL uses 32×32 patches; without cap: 7,040 tokens/sample → OOM on RTX 2060 |
| Processor pixel cap | not applied | `size["longest_edge"] = MAX_PIXELS` | transformers 5.x has no property setter; dict mutation is the only working API |
| Training length control | `max_steps=15000` hardcoded | `--epochs N` → auto-computed from manifest | Hardcoded steps are wrong for datasets other than 30k samples |
| Validation / early stopping | not supported | `--val-manifest` → `EarlyStoppingCallback(patience=3)` | Prevents overfitting; restores best checkpoint automatically |
| Output directory | `asl_lora_output/` (static) | `runs/run_YYYYMMDD_HHMMSS_ep2/` (timestamped) | Keeps each run separate; no accidental overwrites |

### Final token budget

```
8 frames × 480 tokens/frame = 3,840 visual tokens  (1280×720 → 960×512 with 32×32 patches)
+ ~30  chat template
+ ~15  prompt
+ ~30  answer (worst case)
= ~3,915 total  <  MAX_SEQ_LENGTH=5120  ✓  (1,205 tokens headroom)
```
