import torch


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    Generate tokens one at a time.
    Args:
        model         : the GPT model
        idx           : starting token IDs, shape (1, T)
        max_new_tokens: how many tokens to generate
        context_size  : max tokens the model can see at once
        temperature   : 0 = greedy (always picks best token)
                        >0 = random sampling (higher = more creative)
        top_k         : if set, only sample from the top-k best tokens
        eos_id        : stop early if this token appears
    """
    model.eval()

    for _ in range(max_new_tokens):

        # step 1: if input is too long, only keep the last context_size tokens
        idx_cond = idx[:, -context_size:]

        # step 2: run the model to get scores for the next token
        with torch.no_grad():
            logits = model(idx_cond)       # (B, T, vocab_size)

        # step 3: we only care about the very last position
        logits = logits[:, -1, :]          # (B, vocab_size)

        # step 4 (optional): keep only the top_k highest scores, ignore the rest
        if top_k is not None:
            top_vals, _ = torch.topk(logits, top_k)
            cutoff = top_vals[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < cutoff, float("-inf"))

        # step 5: pick the next token
        if temperature == 0.0:
            # greedy: always pick the token with the highest score
            next_id = logits.argmax(dim=-1, keepdim=True)  # (B, 1)
        else:
            # sample: divide by temperature, convert to probabilities, then sample
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)

        # step 6 (optional): stop if the model generated the end-of-sequence token
        if eos_id is not None and (next_id == eos_id).all():
            break

        # step 7: append the new token and loop
        idx = torch.cat([idx, next_id], dim=1)

    return idx
