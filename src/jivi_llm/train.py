import torch
import time
import json
import os
from jivi_llm.loss import calc_loss_batch
from jivi_llm.evaluate import evaluate_model
from jivi_llm.tokenizer import get_tokenizer  # was missing


def generate_sample(model, tokenizer, device, prompt="The", max_new=30):

    model.eval()
    ids = tokenizer.encode(prompt)
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(max_new):
            context_length = model.emb.pos_emb.num_embeddings
            logits = model(x[:, -context_length:])
            next_id = logits[0, -1].argmax().item()
            x = torch.cat([x, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)

    model.train()
    return tokenizer.decode(x[0].tolist())


def train(model, train_loader, val_loader, optimizer, device, num_steps, eval_every=50, log_dir="logs"):

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"run_{int(time.time())}.jsonl")

    tokenizer = get_tokenizer()
    model.to(device)
    step = 0
    tokens_seen = 0
    start = time.time()

    try:
        while step < num_steps:
            for inp, tgt in train_loader:
                if step >= num_steps:
                    break

                model.train()
                optimizer.zero_grad()
                loss = calc_loss_batch(inp, tgt, model, device)
                loss.backward()
                optimizer.step()
                tokens_seen += inp.numel()
                step += 1

                if step % eval_every == 0 or step == 1:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, device, eval_batches=5
                    )
                    sample = generate_sample(model, tokenizer, device)

                    print(f"Step {step:4d} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
                    print(sample)
                    print("-" * 50)

                    # write metrics to log file
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "step": step,
                            "tokens_seen": tokens_seen,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "elapsed_s": round(time.time() - start, 1)
                        }) + "\n")

    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint...")
        torch.save({"step": step, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()},
                   f"checkpoints/interrupt_step{step}.pt")

    print(f"Log saved to: {log_path}")
    return model
