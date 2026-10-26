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

## Model Size

| Config | Params |
|---|---|
| `debug.yaml` | ~13M |
| `gpt_124m.yaml` | ~163M |

## Results

- 19/19 unit tests passing
- Loss decreases from ~10.9 (random) within first few steps
- Greedy generation is fully deterministic (same seed → same output)
- Checkpoints save and resume correctly
