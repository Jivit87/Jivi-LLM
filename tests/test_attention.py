import pytest
import torch
import math
import sys
sys.path.insert(0, "src")

from jivi_llm.config import load_config
from jivi_llm.model import EmbeddingLayer
from jivi_llm.attention import CausalSelfAttention


@pytest.fixture
def cfg():
    return load_config("configs/debug.yaml")


@pytest.fixture
def sample_input(cfg):
    # run a real token batch through the embedding layer
    emb = EmbeddingLayer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.context_length))
    return emb(x)  # (B, T, C)


def test_output_shape(cfg, sample_input):
    # output must be same shape as input (B, T, C)
    attn = CausalSelfAttention(cfg)
    out = attn(sample_input)
    assert out.shape == sample_input.shape


def test_causal_mask(cfg):
    # position 0 must not attend to position 1 (future)
    attn = CausalSelfAttention(cfg)
    assert attn.mask[0, 1] == True   # future blocked
    assert attn.mask[1, 0] == False  # past visible


def test_no_nan_or_inf(cfg, sample_input):
    # forward + backward must produce no NaN or Inf
    attn = CausalSelfAttention(cfg)
    out = attn(sample_input)
    loss = out.sum()
    loss.backward()
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
