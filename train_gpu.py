"""
Train Jivi-LLM on the Gutenberg corpus on a GPU and save checkpoints.

Usage:
    python train_gpu.py --config configs/small_30m.yaml --tokens 100e6 --out checkpoints/small_30m.pt
"""

import argparse
import glob
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from jivi_llm.checkpoint import save_checkpoints
from jivi_llm.config import load_config
from jivi_llm.model import GPTModel
from jivi_llm.tokenizer import get_tokenizer
from jivi_llm.train import get_lr

# whole books held out for validation (never seen in training)
VAL_BOOKS = ["emma", "dracula", "the-verdict", "moby-dick", "treasure-island",
             "heart-of-darkness", "walden", "jane-eyre"]


def load_tokens(cache="data/processed/tokens.pt"):
    if os.path.exists(cache):
        return torch.load(cache)
    tok = get_tokenizer()
    train, val = [], []
    for path in sorted(glob.glob("data/raw/*.txt")):
        name = os.path.basename(path)[:-4]
        if name == "combined":
            continue
        text = open(path, encoding="utf-8", errors="ignore").read()
        (val if name in VAL_BOOKS else train).extend(tok.encode(text, allowed_special="all"))
    train, val = torch.tensor(train), torch.tensor(val)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save((train, val), cache)
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/small_30m.yaml")
    ap.add_argument("--tokens", type=float, default=100e6, help="total training tokens")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--out", default="checkpoints/small_30m.pt")
    ap.add_argument("--save_every", type=int, default=2000, help="steps between checkpoints")
    args = ap.parse_args()

    cfg = load_config(args.config)
    T, bs, device = cfg.context_length, args.batch_size, "cuda"
    train, val = load_tokens()
    print(f"train tokens {len(train):,} | val tokens {len(val):,}")

    torch.manual_seed(0)
    model = GPTModel(cfg).to(device)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M, context {T}")

    # weight decay only on weight matrices, not on LayerNorm / biases
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))

    num_steps = int(args.tokens // (bs * T))
    print(f"{num_steps} steps of {bs}x{T} tokens = {num_steps * bs * T / len(train):.1f} epochs")

    def get_batch(data, gen=None):
        ix = torch.randint(len(data) - T - 1, (bs,), generator=gen)
        x = torch.stack([data[i:i + T] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + T + 1] for i in ix]).to(device)
        return x, y

    def loss_fn(x, y):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))

    @torch.no_grad()
    def evaluate(n_batches=40):
        model.eval()
        gen = torch.Generator().manual_seed(1)  # same batches every time
        out = {name: sum(loss_fn(*get_batch(d, gen)).item() for _ in range(n_batches)) / n_batches
               for name, d in (("train_loss", train), ("val_loss", val))}
        model.train()
        return out

    def save(step, metrics):
        save_checkpoints(args.out, model, optimizer, step, cfg, metrics)

    start, metrics = time.time(), {}
    eval_every = max(1, num_steps // 20)
    for step in range(1, num_steps + 1):
        lr = get_lr(step - 1, num_steps, args.lr)
        for g in optimizer.param_groups:
            g["lr"] = lr
        x, y = get_batch(train)
        loss = loss_fn(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_every == 0 or step == num_steps:
            metrics = evaluate()
            print(f"step {step}/{num_steps} | train {metrics['train_loss']:.3f} | "
                  f"val {metrics['val_loss']:.3f} | lr {lr:.2e} | {time.time() - start:.0f}s", flush=True)
        if step % args.save_every == 0:
            save(step, metrics)

    save(num_steps, metrics)


if __name__ == "__main__":
    main()
