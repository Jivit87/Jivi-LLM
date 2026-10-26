# Jivi-LLM — Improvement Plan

> Ordered from **highest to lowest priority**. Work top to bottom and tick each box when done.
> This file is a plan only. No code changes are described as already made.

---

## Diagnosis (why the model performs badly)

Latest run: `logs/run_1793033304.jsonl`, `configs/debug.yaml` (13M params, context 64, 2 layers, emb 128).

| Fact | Value |
|---|---|
| Corpus (`data/raw/combined.txt`) | 3.4 MB, about **914K tokens** |
| Tokens seen in training | 500 steps × 512 = **256K tokens** (about 0.28 epochs) |
| Loss | 10.95 → 5.59 (train), 5.59 (val), still falling |

A loss near 5.5 is roughly the level of a model that only knows word frequencies. It has not learned grammar yet. The model is **under-trained**, with a **small config** and **small data**. The pipeline itself works.

Rule of thumb: a model needs about 20 tokens per parameter. The 124M model would need about 2.5B tokens; we have 0.9M.

---

## Priority 1 — HIGH (biggest impact, do first)

### 1.1 Train much longer on the existing corpus
**Why:** only 0.28 epochs have been seen. This is the main reason the output is bad.
**What:** run the debug model for 10–20+ epochs (about 10M+ tokens, around 20,000–35,000 steps at batch 8 × 64). Loss should fall well below 5, and samples should start to look like English.
- [ ] Decide a step or epoch budget (e.g. 20 epochs)
- [ ] Run training with periodic checkpoints (every ~1000 steps)
- [ ] Confirm val loss keeps falling (watch for the point where val rises while train falls = overfitting)
- [ ] Record final train/val loss and a sample in the log

### 1.2 Learning-rate schedule + gradient clipping (`train.py`)
**Why:** a constant lr of 1e-3 with no warmup is unstable early and sub-optimal late.
**What:**
- [ ] Linear warmup for the first ~1–5% of steps
- [ ] Cosine decay to ~10% of peak lr
- [ ] Gradient clipping (`clip_grad_norm_`, max norm 1.0)
- [ ] Log the current lr with every metric line (the plan already requires this)

### 1.3 Use a bigger "small" model config
**Why:** `debug.yaml` (context 64, 2 layers) is a pipeline test config. It is too small for fluent text.
**What:** add a separate config (do not reuse the debug name), for example `configs/small_30m.yaml`:
- [ ] context_length 256, n_layers 6, n_heads 6, emb_dim 384 (about 30–40M params)
- [ ] Verify it fits in memory on the 8 GB Mac with a conservative batch size
- [ ] Check that the config name clearly says it is not the 124M target

### 1.4 Fix sampling in training-time samples
**Why:** `generate_sample` in `train.py` is hardcoded to greedy (argmax). Under-trained models loop ("the the the…"), which makes quality look worse than it is.
**What:**
- [ ] Reuse `generate()` from `generate.py` instead of the duplicate loop
- [ ] Log one greedy sample and one sample with `temperature≈0.8, top_k≈40`
- [ ] Use a fixed seed so samples are comparable across checkpoints
- [ ] Use a longer prompt and more new tokens (e.g. 60+)

---

## Priority 2 — MEDIUM (data quality and trustworthy measurement)

### 2.1 Clean the corpus
**Why:** `combined.txt` contains Project Gutenberg licence headers and footers (12 "PROJECT GUTENBERG" matches). The model learns to generate boilerplate.
- [ ] Strip text before `*** START OF` and after `*** END OF` in each book
- [ ] Normalize repeated blank lines and odd whitespace
- [ ] Re-count tokens and record the number in `data/README.md`
- [ ] Update `data/README.md` (it still describes only *The Verdict*) with all sources, licences and token counts

### 2.2 Better train/validation split
**Why:** a 90/10 cut of the combined text means validation is just the end of the last book, which is not representative.
- [ ] Split per book, or hold out a whole book for validation (and use random chunks from each book as a second check)
- [ ] Make sure no validation text appears in training (dedupe)
- [ ] Keep the split fixed and documented

### 2.3 More reliable evaluation
**Why:** `evaluate_model` averages only 5 random batches, so the numbers are noisy.
- [ ] Evaluate on a fixed, larger set (e.g. 50–100 batches, or the whole val set)
- [ ] Use `shuffle=False` for the val loader
- [ ] Also log perplexity (`exp(val_loss)`)

### 2.4 Add the missing checkpoint/resume wiring
**Why:** a long run must survive interruption.
- [ ] Save checkpoints from `train()` itself (model, optimizer, step, config, tokens seen, metrics, tokenizer name)
- [ ] Add a `--resume` path and log which file was resumed
- [ ] Also store RNG state and the lr-scheduler step (the plan lists it)
- [ ] Keep only the last N checkpoints to avoid filling disk (each is about 160 MB)

### 2.5 Fix `generate_text.py` config mismatch
**Why:** `--config` defaults to `debug.yaml`, so loading a different-size checkpoint fails or gives junk.
- [ ] Read the config stored inside the checkpoint instead of trusting the CLI default
- [ ] Add `--seed` for reproducible sampling
- [ ] Add a `--device` option (CPU / MPS), reporting the device used

---

## Priority 3 — MEDIUM-LOW (model quality tricks)

### 3.1 Tie input embedding and output head weights
**Why:** GPT-2 shares the token embedding with the output layer. It saves about 38M params at 124M size, and generally helps small models.
- [ ] Set `lm_head.weight = tok_emb.weight`
- [ ] Update the parameter-count test (the count would fall from about 163M to about 124M, matching the plan's target)
- [ ] Verify the checkpoint save and load still works

### 3.2 GPT-2 style weight initialization
- [ ] Normal init (std 0.02) for Linear and Embedding layers
- [ ] Scale residual-output projections (`W_o`, FFN second layer) by `1/sqrt(2 * n_layers)`
- [ ] Check that initial loss is close to `ln(50257) ≈ 10.82`

### 3.3 Weight decay only on weight matrices
- [ ] Put LayerNorm scale/shift and biases in a no-decay param group
- [ ] Apply weight decay (0.1) only to 2-D weights
- [ ] Use explicit `betas=(0.9, 0.95)`, matching the README example

### 3.4 Tune dropout
**Why:** with a small corpus, overfitting is the risk, but for a short run on a larger corpus, dropout can slow learning.
- [ ] Try 0.1 vs 0.0 for the longer run and compare val loss
- [ ] Keep the better value in the config

---

## Priority 4 — LOW (scale-up, efficiency, polish)

### 4.1 Efficiency
- [ ] Use `F.scaled_dot_product_attention` (fused causal attention) instead of the manual mask
- [ ] Use mixed precision (`torch.autocast`, bf16) on the GPU
- [ ] Add gradient accumulation to get larger effective batches on 8 GB
- [ ] Benchmark tokens/second and peak memory, then record them
- [ ] Tokenize once and cache the IDs in `data/processed/` instead of re-encoding on every run

### 4.2 Move to the real 124M run (RTX 4090)
**Only after** priorities 1–3 are done and the small model clearly learns.
- [ ] Gather a much larger legal corpus (hundreds of millions of tokens, e.g. OpenWebText or similar)
- [ ] Document source, licence, size, dedup and split
- [ ] Verify CUDA/PyTorch compatibility, VRAM and disk, and set up remote checkpoint backup
- [ ] Benchmark first, then pick a token budget based on measured throughput
- [ ] Be honest in the write-up about what the model can and cannot do

### 4.3 Housekeeping
- [ ] Update the stale status line at the bottom of `task.md`
- [ ] Remove `PYTHONPATH=src` from the README test command (the editable install makes it unnecessary)
- [ ] Remove the leftover `sys.path.insert(0, "src")` from `generate_text.py`
- [ ] Clean old test checkpoints (`ckpt_step0.pt`, `ckpt_step5.pt`, `test_ckpt.pt`)
- [ ] Add tests for any new behaviour (scheduler, weight tying, per-group weight decay, checkpoint resume)
- [ ] Commit the untracked files (`tokenizer.py`, `test_tokenizer.py`, `pyproject.toml`) and the deleted `src/__init__.py`

---

## Suggested execution order

1. 1.2 → 1.4 → 1.1: scheduler and clipping, better samples, then a long run on the current setup. This gives a quick win and a baseline.
2. 2.1 → 2.2 → 2.3: clean data, fix split, steadier evaluation.
3. 1.3 + 3.x: bigger config and model tricks, compared against the baseline.
4. 2.4 / 2.5, then 4.x: resume wiring, tooling, then the scale-up.

## How to know it worked

- [ ] Val loss clearly below the current 5.59 (target: under 4.5 on the debug model, lower on a bigger one)
- [ ] Samples with sampling enabled read as mostly grammatical English
- [ ] Train and val loss curves logged and compared across runs
- [ ] No sign of memorization (val loss not rising while train loss falls)
