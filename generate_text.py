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

from jivi_llm.config import load_config
from jivi_llm.model import GPTModel
from jivi_llm.tokenizer import get_tokenizer, encode, decode
from jivi_llm.generate import generate


def main():
    parser = argparse.ArgumentParser(description="Generate text with Jivi-LLM")
    parser.add_argument("prompt",                   type=str,   help="Starting text")
    parser.add_argument("--max_new",                type=int,   default=50,    help="Number of tokens to generate")
    parser.add_argument("--temperature",            type=float, default=0.0,   help="0 = greedy, >0 = creative")
    parser.add_argument("--top_k",                  type=int,   default=None,  help="Only sample from top-k tokens")
    parser.add_argument("--config",                 type=str,   default="configs/debug.yaml")
    parser.add_argument("--checkpoint",             type=str,   default=None,  help="Path to checkpoint .pt file")
    args = parser.parse_args()

    # load config and model
    cfg = load_config(args.config)
    model = GPTModel(cfg)

    # optionally load trained weights from a checkpoint
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
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
