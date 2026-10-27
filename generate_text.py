"""
Usage:
    python generate_text.py "Your prompt here"
    python generate_text.py "The quick brown" --max_new 50 --temperature 0.8 --top_k 20
    python generate_text.py "Once upon a time" --checkpoint checkpoints/ckpt_step5.pt
"""

import torch
import sys
import argparse
sys.path.insert(0, "src")

from jivi_llm.config import load_config, GPTConfig
from jivi_llm.model import GPTModel
from jivi_llm.tokenizer import get_tokenizer, encode, decode
from jivi_llm.generate import generate


def main():
    parser = argparse.ArgumentParser(description="Generate text with Jivi-LLM")
    parser.add_argument("prompt",                   type=str,   help="Starting text")
    parser.add_argument("--max_new",                type=int,   default=50,    help="Number of tokens to generate")
    parser.add_argument("--temperature",            type=float, default=0.0,   help="0 = greedy, >0 = creative")
    parser.add_argument("--top_k",                  type=int,   default=None,  help="Only sample from top-k tokens")
    parser.add_argument("--config",                 type=str,   default=None,  help="Config YAML (only needed if the checkpoint has no stored config)")
    parser.add_argument("--checkpoint",             type=str,   default=None,  help="Path to checkpoint .pt file")
    args = parser.parse_args()

    # load the checkpoint first: it carries the config it was trained with
    ckpt = torch.load(args.checkpoint, map_location="cpu") if args.checkpoint else None

    if ckpt is not None and ckpt.get("config"):
        cfg = GPTConfig(**ckpt["config"])
        print(f"Using config stored in checkpoint (emb_dim={cfg.emb_dim}, n_layers={cfg.n_layers}, context={cfg.context_length})")
    else:
        cfg = load_config(args.config or "configs/debug.yaml")
        print(f"Using config file: {args.config or 'configs/debug.yaml'}")

    model = GPTModel(cfg)

    if ckpt is not None:
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint: {args.checkpoint} (step {ckpt['step']})")
    else:
        print("No checkpoint provided — using random weights")

    # tokenize the prompt
    tok = get_tokenizer()
    ids = encode(args.prompt, tok)
    idx = torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # (1, T)

    # generate
    print(f"\nPrompt: {args.prompt}")
    print("-" * 40)
    out = generate(
        model, idx,
        max_new_tokens=args.max_new,
        context_size=cfg.context_length,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(decode(out[0].tolist(), tok))


if __name__ == "__main__":
    main()
