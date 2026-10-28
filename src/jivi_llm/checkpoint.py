import torch
import os

def save_checkpoints(path, model, optimizer, step, config, metrics=None, tokenizer="gpt2"):
    # saving everything needed to resume training 
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": vars(config),
        "metrics": metrics or {},
        "tokenizer": tokenizer  # tiktoken name, or path to a custom tokenizer.json
    }, path)

    print(f"Checkpoint saved: {path}")


def load_checkpoints(path, model, optimizer):
    # load weights and optimizer state, return metadata
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"Resumed from: {path} (step {checkpoint['step']})")
    return checkpoint # caller can read step, config, metrics

    

    