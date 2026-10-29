"""
Generate text from a trained medical model checkpoint.

Usage:
    python generate_med.py --prompt "Metformin is" --max-new 80 --temperature 0.7 --top-k 50
    python generate_med.py                       # runs a built-in set of prompts
NOT medical advice: the model is small and often states things that are false.
"""
import argparse
import sys

import torch

sys.path.insert(0, "src")
from jivi_llm.config import GPTConfig  # noqa: E402
from jivi_llm.generate import generate  # noqa: E402
from jivi_llm.model import GPTModel  # noqa: E402
from jivi_llm.tokenizer import get_tokenizer  # noqa: E402

DEFAULT_PROMPTS = [
    "Metformin is",
    "The most common symptoms of type 2 diabetes include",
    "A 45-year-old woman presents with chest pain radiating to the left arm. The initial evaluation should include",
    "Hypertension is defined as",
    "Treatment of community-acquired pneumonia in adults",
    "Aspirin should not be given to children because",
    "Patients taking warfarin should avoid",
    "The capital of France is",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/med124m_v1/latest.pt")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.7, help="0 = greedy")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    cfg = GPTConfig(**ck["config"])
    tok = get_tokenizer(ck["tokenizer"])
    model = GPTModel(cfg)
    model.load_state_dict(ck["model_state"])
    model = model.cuda().eval()
    print(f"loaded {a.ckpt} (step {ck['step']}, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params)\n")

    torch.manual_seed(a.seed)
    for p in ([a.prompt] if a.prompt else DEFAULT_PROMPTS):
        ids = torch.tensor([tok.encode(p)], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = generate(model, ids, a.max_new, cfg.context_length, temperature=a.temperature,
                           top_k=a.top_k if a.temperature > 0 else None, eos_id=tok.eot_token)
        print(f"PROMPT: {p}\n  -> {tok.decode(out[0].tolist())}\n")


if __name__ == "__main__":
    main()
