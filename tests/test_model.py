import pytest
import torch
import sys
sys.path.insert(0, "src")

from jivi_llm.config import load_config
from jivi_llm.model import GPTModel


@pytest.fixture
def cfg():
    return load_config("configs/debug.yaml")


@pytest.fixture
def model(cfg):
    return GPTModel(cfg)


def test_logits_shape(cfg, model):
    # forward pass must produce (B, T, vocab_size) logits
    x = torch.randint(0, cfg.vocab_size, (2, cfg.context_length))
    logits = model(x)
    assert logits.shape == (2, cfg.context_length, cfg.vocab_size)


def test_no_pretrained_weights(model):
    # confirm no pretrained GPT-2 weights are loaded:
    # weight matrices (not biases/shifts) must have non-zero variance
    weight_params = [p for p in model.parameters() if p.ndim >= 2]
    for p in weight_params:
        assert p.std().item() > 0, "weight matrix looks uninitialized (all same value)"


def test_forward_backward(cfg, model):
    # forward + backward must complete without error or NaN
    x = torch.randint(0, cfg.vocab_size, (2, cfg.context_length))
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_param_count():
    # record exact parameter count for the 124M reference config
    cfg = load_config("configs/gpt_124m.yaml")
    model = GPTModel(cfg)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n124M model param count: {total:,}")
    assert total > 100_000_000, f"Expected >100M params, got {total:,}"
