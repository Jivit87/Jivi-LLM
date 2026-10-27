# Jivi-LLM

A GPT-2 style language model built from scratch in PyTorch.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/ -v
```

## Train

```python
from jivi_llm.config import load_config
from jivi_llm.model import GPTModel
from jivi_llm.tokenizer import get_tokenizer, encode
from jivi_llm.dataset import create_dataloader
from jivi_llm.train import train
import torch

cfg = load_config("configs/debug.yaml")
tok = get_tokenizer()
with open("data/raw/the-verdict.txt") as f:
    ids = encode(f.read(), tok)

split = int(len(ids) * 0.9)
train_loader = create_dataloader(ids[:split], cfg.context_length, stride=cfg.context_length, batch_size=4)
val_loader   = create_dataloader(ids[split:], cfg.context_length, stride=cfg.context_length, batch_size=4)

model = GPTModel(cfg)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95))
train(model, train_loader, val_loader, optimizer, device="cpu", num_steps=100, eval_every=10)
```

## Generate

```python
from jivi_llm.generate import generate
from jivi_llm.tokenizer import get_tokenizer, encode, decode
import torch

tok = get_tokenizer()
ids = torch.tensor(encode("The quick brown", tok)).unsqueeze(0)
out = generate(model, ids, max_new_tokens=30, context_size=cfg.context_length, temperature=0.8, top_k=20)
print(decode(out[0].tolist(), tok))
```

## Architecture

| Component | File |
|---|---|
| Config loader | `src/jivi_llm/config.py` |
| Tokenizer (GPT-2 BPE) | `src/jivi_llm/tokenizer.py` |
| Dataset + DataLoader | `src/jivi_llm/dataset.py` |
| Embeddings | `src/jivi_llm/model.py` |
| Single/Multi-Head Attention | `src/jivi_llm/attention.py` |
| LayerNorm + FeedForward | `src/jivi_llm/model.py` |
| Transformer Block | `src/jivi_llm/transformer_block.py` |
| Full GPT Model | `src/jivi_llm/model.py` |
| Loss | `src/jivi_llm/loss.py` |
| Evaluate | `src/jivi_llm/evaluate.py` |
| Training Loop | `src/jivi_llm/train.py` |
| Checkpoint | `src/jivi_llm/checkpoint.py` |
| Generation | `src/jivi_llm/generate.py` |

## Data

`python download_data.py` downloads 115 public-domain books from Project Gutenberg, strips the licence header/footer from each, and writes `data/raw/combined.txt`.

| | |
|---|---|
| Combined corpus | ~94 MB, ~25.6M GPT-2 tokens |
| Training set | 107 books, 24.6M tokens |
| Validation set | 8 whole books held out (*Emma*, *Dracula*, *The Verdict*, *Moby Dick*, *Treasure Island*, *Heart of Darkness*, *Walden*, *Jane Eyre*), 1.48M tokens |

Validation is by whole book, so it measures generalization to unseen text. See `data/README.md` for details.

## Model Size

| Config | Params | Context |
|---|---|---|
| `debug.yaml` | ~13M | 64 |
| `small_30m.yaml` | ~49M (measured; input and output embeddings are not tied) | 256 |
| `gpt_124m.yaml` | ~163M (~124M once embeddings are tied) | 1024 |

`small_30m.yaml` is named for the transformer body; the untied 50257-token embedding and output head add most of the parameters.

## GPU Experiments

Trained on one NVIDIA RTX PRO 4000 Blackwell (24 GB) with bf16 autocast, AdamW (peak lr 6e-4, betas 0.9/0.95, weight decay 0.1 on weight matrices only), linear warmup + cosine decay, and gradient clipping at 1.0. Validation loss is on the 8 held-out books.

| Config | Params | Tokens seen | Epochs | Time | Train loss | Val loss | Val perplexity | Speed | Peak memory |
|---|---|---|---|---|---|---|---|---|---|
| `small_30m` | 49.3M | 30M | 1.2 | 6 min | 4.12 | 4.53 | ~93 | 79k tok/s | 7.4 GB |
| `gpt_124m` | 163M | 30M | 1.2 | 32 min | 3.77 | 4.28 | ~72 | 15.6k tok/s | 20.3 GB |
| `small_30m` | 49.3M | 100M | 4.1 | 20 min | 3.63 | **4.16** | ~64 | 84k tok/s | 7.4 GB |

The previous baseline (13M debug model, 0.28 epochs, constant lr) reached a loss of about 5.59.

**Takeaways**
- At equal tokens (30M) the 124M config beats the 49M config (val 4.28 vs 4.53), at about 5x the training time.
- The small model trained for 100M tokens beats the 124M trained for 30M, and takes about 20 minutes. With this corpus, more passes over the data matter more than model size.
- Neither model overfits yet: the train/val gap is about 0.4-0.5 in all runs. The 124M config would start to overfit after a few more epochs, because 25M tokens is far below the ~2.5B (about 20 tokens per parameter) it needs.
- Samples (prompt "It was a dark and stormy night, and"): sampled text (temperature 0.8, top-k 40) is grammatical English with dialogue, e.g. "I felt as if I had been living under little trees, but I had not seen them", but it drifts and is often meaningless. Greedy decoding still loops ("I have been a little more than a little while...").
- The 124M run needed 20 GB because attention builds the full causal mask. Fused attention (`F.scaled_dot_product_attention`) is the next efficiency step.

These numbers come from throwaway comparison scripts, not from `train.py` itself. They are single runs with one seed, so small differences are not conclusive.

## Results

- 19/19 unit tests passing
- Loss decreases from ~10.9 (random) within first few steps
- Greedy generation is fully deterministic (same seed → same output)
- Checkpoints save and resume correctly
