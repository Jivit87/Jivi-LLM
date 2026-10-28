"""
Train the medical-domain GPT (configs/med_124m.yaml) on the tokenized medical corpus.

Data: data/medical/tokenized/<source>/{train,val}.bin (uint16 memmaps, see data_pipeline/10_tokenize.py).
Sampling: each source is cut into non-overlapping windows; every epoch visits each window once in a fresh shuffled order,
and each sequence in a batch picks its source with probability proportional to --weights (default: proportional to size,
so every source gets about the same number of epochs). The sampler state is saved in checkpoints, so resuming is exact.
LR schedule: linear warmup, flat, then a cosine decay over the last --decay-frac of steps (a pure function of the step,
so resuming with a smaller --tokens simply moves the decay earlier: you can decide after any epoch to wind down).

Usage:
    python train_med.py --name pilot --tokens 30e6 --lr 6e-4
    python train_med.py --name main --tokens 1.5e9 --dropout 0.0 --compile max
    python train_med.py --name main --tokens 1.5e9 --dropout 0.0 --compile max --resume    # continue from latest.pt
Stop cleanly at any time with:  touch checkpoints/<name>/STOP   (saves a checkpoint, then exits)
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from jivi_llm.checkpoint import load_checkpoints, save_checkpoints  # noqa: E402
from jivi_llm.config import load_config  # noqa: E402
from jivi_llm.generate import generate  # noqa: E402
from jivi_llm.model import GPTModel  # noqa: E402
from jivi_llm.tokenizer import get_tokenizer  # noqa: E402

DATA = "data/medical/tokenized"
TOKENIZER = "data/medical/tokenizer/med_bpe_32k.json"
PROMPTS = ["Metformin is", "The most common cause of community-acquired pneumonia is",
           "In patients with chronic kidney disease,", "Once upon a time"]


# ---------------------------------------------------------------- data
class MixedSampler:
    """Epoch-exact, resumable sampling of context-length windows from several token files."""

    def __init__(self, T, weights, seed=0):
        meta = json.load(open(os.path.join(DATA, "meta.json")))
        self.T, self.seed = T, seed
        self.names = [n for n in meta["sources"] if meta["sources"][n]["train_tokens"] > 0]
        self.arr = {n: np.memmap(os.path.join(DATA, n, "train.bin"), dtype=np.uint16, mode="r") for n in self.names}
        self.n_chunks = {n: (len(self.arr[n]) - 1) // T for n in self.names}
        w = np.array([weights.get(n, self.n_chunks[n]) for n in self.names], dtype=np.float64)
        self.p = w / w.sum()
        self.state = {n: {"epoch": 0, "pos": 0} for n in self.names}
        self.batches = 0
        self._perm = {}

    def _order(self, n):
        e = self.state[n]["epoch"]
        if self._perm.get(n, (None,))[0] != e:
            rng = np.random.default_rng([self.seed, self.names.index(n), e])
            self._perm[n] = (e, rng.permutation(self.n_chunks[n]))
        return self._perm[n][1]

    def next_batch(self, B):
        rng = np.random.default_rng([self.seed, 1_000_003, self.batches])
        self.batches += 1
        src = rng.choice(len(self.names), size=B, p=self.p)
        xs, ys = [], []
        for si in src:
            n = self.names[si]
            st = self.state[n]
            if st["pos"] >= self.n_chunks[n]:
                st["epoch"] += 1
                st["pos"] = 0
            start = int(self._order(n)[st["pos"]]) * self.T
            st["pos"] += 1
            w = self.arr[n][start:start + self.T + 1].astype(np.int64)
            xs.append(w[:-1])
            ys.append(w[1:])
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))

    def epochs(self):
        return {n: round(self.state[n]["epoch"] + self.state[n]["pos"] / self.n_chunks[n], 2) for n in self.names}

    def state_dict(self):
        return {"state": self.state, "batches": self.batches}

    def load_state_dict(self, d):
        self.state, self.batches = d["state"], d["batches"]


def load_val(T, n_seq=128):
    """Fixed, evenly spaced validation windows per source."""
    val = {}
    for n in sorted(os.listdir(DATA)):
        path = os.path.join(DATA, n, "val.bin")
        if not os.path.exists(path):
            continue
        arr = np.memmap(path, dtype=np.uint16, mode="r")
        chunks = (len(arr) - 1) // T
        if chunks < 1:
            continue
        idx = np.linspace(0, chunks - 1, num=min(n_seq, chunks), dtype=int)
        w = np.stack([arr[i * T:i * T + T + 1].astype(np.int64) for i in idx])
        val[n] = (torch.from_numpy(w[:, :-1]), torch.from_numpy(w[:, 1:]))
    return val


# ---------------------------------------------------------------- model helpers
def init_weights(model, cfg):
    """GPT-2 style init (std 0.02, residual output projections scaled by 1/sqrt(2*layers)).
    The repo's default PyTorch init (Embedding ~ N(0,1)) is far too large once the output head shares the embedding."""
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, "bias", None) is not None:
                torch.nn.init.zeros_(m.bias)
    scale = 0.02 / math.sqrt(2 * cfg.n_layers)
    for name, p in model.named_parameters():
        if name.endswith("attn.W_o.weight") or name.endswith("ff.net.2.weight"):
            torch.nn.init.normal_(p, mean=0.0, std=scale)


def lr_at(step, total, peak, warmup, decay_frac, min_ratio=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup
    decay_start = int(total * (1 - decay_frac))
    if step < decay_start:
        return peak
    t = min(1.0, (step - decay_start) / max(1, total - decay_start))
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


def loss_fn(model, x, y):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x)
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))


@torch.no_grad()
def evaluate(model, val, bs=16):
    model.eval()
    out = {}
    for n, (x, y) in val.items():
        tot = 0.0
        for i in range(0, len(x), bs):
            xb, yb = x[i:i + bs].cuda(non_blocking=True), y[i:i + bs].cuda(non_blocking=True)
            tot += loss_fn(model, xb, yb).item() * len(xb)
        out[n] = tot / len(x)
    model.train()
    return out


def sample_text(model, tok, cfg, seed=0):
    model.eval()
    torch.manual_seed(seed)
    outs = []
    for p in PROMPTS:
        ids = torch.tensor([tok.encode(p)], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            g = generate(model, ids, 40, cfg.context_length, temperature=0.8, top_k=40)
        outs.append(tok.decode(g[0].tolist()).replace("\n", " "))
    model.train()
    return outs


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/med_124m.yaml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--tokens", type=float, required=True, help="total training tokens")
    ap.add_argument("--micro-bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8, help="micro-batches per optimizer step")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--decay-frac", type=float, default=0.2)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=None, help="override config dropout")
    ap.add_argument("--compile", choices=["none", "default", "max"], default="max")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--sample-every", type=int, default=4, help="generate text every N evals")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--weights", nargs="*", default=[], help="source=weight overrides, e.g. pmc_oa=300 guidelines=40")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.dropout is not None:
        cfg.dropout = a.dropout
    T = cfg.context_length
    tok_per_step = a.micro_bs * a.accum * T
    total_steps = int(a.tokens // tok_per_step)
    ckpt_dir = os.path.join("checkpoints", a.name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs("logs/med", exist_ok=True)
    log_path = f"logs/med/{a.name}.jsonl"
    stop_path = os.path.join(ckpt_dir, "STOP")
    if os.path.exists(stop_path):
        os.remove(stop_path)

    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    weights = {k: float(v) for k, v in (kv.split("=") for kv in a.weights)}
    sampler = MixedSampler(T, weights, seed=a.seed)
    val = load_val(T)
    tok = get_tokenizer(TOKENIZER)

    model = GPTModel(cfg).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=True)
    step = 0
    if a.resume:
        ck = load_checkpoints(os.path.join(ckpt_dir, "latest.pt"), model, opt)
        step = ck["step"]
        sampler.load_state_dict(ck["metrics"]["sampler"])
    else:
        init_weights(model, cfg)
    run = model if a.compile == "none" else torch.compile(
        model, mode="max-autotune-no-cudagraphs" if a.compile == "max" else None)

    print(f"{a.name}: {n_params/1e6:.2f}M params | {tok_per_step:,} tokens/step | {total_steps:,} steps | "
          f"{total_steps*tok_per_step/1e6:.0f}M tokens | dropout {cfg.dropout} | lr {a.lr} | start step {step}", flush=True)
    print("sampling probabilities:", {n: round(float(p), 3) for n, p in zip(sampler.names, sampler.p)}, flush=True)

    def log(rec):
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def save(tag_step, metrics):
        metrics = {**metrics, "sampler": sampler.state_dict()}
        path = os.path.join(ckpt_dir, f"step_{tag_step:06d}.pt")
        save_checkpoints(path, model, opt, tag_step, cfg, metrics, tokenizer=TOKENIZER)
        tmp = os.path.join(ckpt_dir, "latest.pt.tmp")
        os.system(f"cp '{path}' '{tmp}' && mv '{tmp}' '{os.path.join(ckpt_dir, 'latest.pt')}'")

    model.train()
    last_val, n_eval, t_log, tok_log = {}, 0, time.time(), 0
    t_start = time.time()
    while step < total_steps:
        lr = lr_at(step, total_steps, a.lr, a.warmup, a.decay_frac, a.min_lr_ratio)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_acc = 0.0
        for _ in range(a.accum):
            x, y = sampler.next_batch(a.micro_bs)
            loss = loss_fn(run, x.cuda(non_blocking=True), y.cuda(non_blocking=True)) / a.accum
            loss.backward()
            loss_acc += loss.item()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        tok_log += tok_per_step

        if step % a.log_every == 0:
            torch.cuda.synchronize()
            dt = time.time() - t_log
            rec = {"step": step, "tokens": step * tok_per_step, "loss": round(loss_acc, 4), "lr": lr,
                   "grad_norm": round(gnorm, 3), "tok_s": int(tok_log / dt), "elapsed_h": round((time.time() - t_start) / 3600, 3)}
            log(rec)
            print(f"step {step}/{total_steps} loss {loss_acc:.3f} lr {lr:.2e} gnorm {gnorm:.2f} {rec['tok_s']} tok/s "
                  f"| {rec['elapsed_h']:.2f}h", flush=True)
            t_log, tok_log = time.time(), 0
            if not math.isfinite(loss_acc):
                print("loss is NaN/inf, stopping", flush=True)
                break

        if step % a.eval_every == 0 or step == total_steps:
            last_val = evaluate(model, val)
            n_eval += 1
            allv = sum(last_val[n] * float(p) for n, p in zip(sampler.names, sampler.p) if n in last_val)
            rec = {"step": step, "val": {k: round(v, 4) for k, v in last_val.items()}, "val_mix": round(allv, 4),
                   "epochs": sampler.epochs()}
            log(rec)
            print(f"  VAL step {step}: mix {allv:.3f} | " + " ".join(f"{k} {v:.3f}" for k, v in last_val.items()) +
                  f" | epochs {sampler.epochs()}", flush=True)
            if n_eval % a.sample_every == 0 or step == total_steps:
                for s in sample_text(model, tok, cfg):
                    print("   >", s[:300], flush=True)
        if step % a.save_every == 0 or step == total_steps:
            save(step, {"val": last_val})
        if os.path.exists(stop_path):
            print("STOP file found: saving and exiting", flush=True)
            save(step, {"val": last_val})
            break
    print(f"done at step {step}, {(time.time()-t_start)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
