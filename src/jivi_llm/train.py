import math
import torch
import time
import json
import os
from jivi_llm.loss import calc_loss_batch
from jivi_llm.evaluate import evaluate_model
from jivi_llm.tokenizer import get_tokenizer 
from jivi_llm.generate import generate


def generate_sample(model, tokenizer, device, prompt="It was a dark and stormy night, and", max_new=60, seed=123):
    """Returns (greedy_text, sampled_text). The sampled one uses a fixed seed so
    samples are comparable across checkpoints, without disturbing the global RNG."""
    context_size = model.emb.pos_emb.num_embeddings
    ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

    greedy = generate(model, ids, max_new, context_size, temperature=0.0)

    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    sampled = generate(model, ids, max_new, context_size, temperature=0.8, top_k=40)
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)

    model.train()  # generate() leaves the model in eval mode
    return tokenizer.decode(greedy[0].tolist()), tokenizer.decode(sampled[0].tolist())


def get_lr(step, num_steps, peak_lr, warmup_frac=0.02, min_ratio=0.1):
    # Gradually increase learning rate during warmup
    warmup_steps = max(1, int(num_steps * warmup_frac))

    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps

    # Calculate progress through cosine decay (0 to 1)
    progress = (step - warmup_steps) / max(1, num_steps - warmup_steps)
    progress = min(progress, 1.0)

    # Gradually decrease learning rate to its minimum
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return peak_lr * (min_ratio + (1 - min_ratio) * cosine)


def train(model, train_loader, val_loader, optimizer, device, num_steps, eval_every=50, log_dir="logs"):

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"run_{int(time.time())}.jsonl")

    tokenizer = get_tokenizer()
    model.to(device)
    step = 0
    tokens_seen = 0
    start = time.time()
    peak_lr = optimizer.param_groups[0]["lr"]  # read once, before any decay

    try:
        while step < num_steps:
            for inp, tgt in train_loader:
                if step >= num_steps:
                    break

                model.train()
                lr = get_lr(step, num_steps, peak_lr)
                for g in optimizer.param_groups:
                    g["lr"] = lr
                optimizer.zero_grad()
                loss = calc_loss_batch(inp, tgt, model, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tokens_seen += inp.numel()
                step += 1

                if step % eval_every == 0 or step == 1:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, device, eval_batches=5
                    )
                    greedy_text, sampled_text = generate_sample(model, tokenizer, device)

                    print(f"Step {step:4d} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | LR: {lr:.2e}")
                    print("[greedy]  ", greedy_text)
                    print("[sampled] ", sampled_text)
                    print("-" * 50)

                    # write metrics to log file
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "step": step,
                            "tokens_seen": tokens_seen,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "lr": lr,
                            "greedy_sample": greedy_text,
                            "sampled_sample": sampled_text,
                            "elapsed_s": round(time.time() - start, 1)
                        }) + "\n")

    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint...")
        torch.save({"step": step, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()},
                   f"checkpoints/interrupt_step{step}.pt")

    print(f"Log saved to: {log_path}")
    return model
