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

from jivi_llm.attention import MultiHeadAttention


def test_mha_output_shape(cfg, sample_input):
    # multi-head output must preserve (B, T, C)
    mha = MultiHeadAttention(cfg)
    out = mha(sample_input)
    assert out.shape == sample_input.shape


def test_mha_gradients(cfg, sample_input):
    # gradients must reach all 4 projection matrices
    mha = MultiHeadAttention(cfg)
    out = mha(sample_input)
    out.sum().backward()
    for name in ["W_q", "W_k", "W_v", "W_o"]:
        assert getattr(mha, name).weight.grad is not None


def test_mha_no_nan_or_inf(cfg, sample_input):
    # forward pass must be numerically clean
    mha = MultiHeadAttention(cfg)
    out = mha(sample_input)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()



# ---- fused attention (scaled_dot_product_attention) must match the original manual math ----

def _manual_mha_forward(mha, x):
    # the ORIGINAL implementation (now kept as a comment in attention.py), used here as the reference
    B, T, C = x.shape
    H, d = mha.n_heads, mha.d_head
    Q = mha.W_q(x).view(B, T, H, d).transpose(1, 2)
    K = mha.W_k(x).view(B, T, H, d).transpose(1, 2)
    V = mha.W_v(x).view(B, T, H, d).transpose(1, 2)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d)
    scores = scores.masked_fill(mha.mask[:T, :T], float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    out = (weights @ V).transpose(1, 2).contiguous().view(B, T, C)
    return mha.W_o(out)


def test_fused_matches_manual_attention(cfg, sample_input):
    torch.manual_seed(0)
    mha = MultiHeadAttention(cfg).eval()
    assert torch.allclose(mha(sample_input), _manual_mha_forward(mha, sample_input), atol=1e-5)


def test_fused_matches_manual_gradients(cfg, sample_input):
    torch.manual_seed(0)
    mha = MultiHeadAttention(cfg).eval()
    x1 = sample_input.detach().clone().requires_grad_(True)
    x2 = sample_input.detach().clone().requires_grad_(True)
    mha(x1).sum().backward()
    g_fused = [p.grad.clone() for p in mha.parameters()]
    mha.zero_grad()
    _manual_mha_forward(mha, x2).sum().backward()
    for a, b in zip(g_fused, [p.grad for p in mha.parameters()]):
        assert torch.allclose(a, b, atol=1e-4)
    assert torch.allclose(x1.grad, x2.grad, atol=1e-4)


def test_fused_works_for_sequences_shorter_than_context(cfg):
    mha = MultiHeadAttention(cfg).eval()
    x = torch.randn(2, cfg.context_length // 2, cfg.emb_dim)
    assert torch.allclose(mha(x), _manual_mha_forward(mha, x), atol=1e-5)


def test_fused_is_causal(cfg):
    # changing a future token must not change earlier positions
    mha = MultiHeadAttention(cfg).eval()
    x = torch.randn(1, 8, cfg.emb_dim)
    y = x.clone()
    y[:, -1] += 5.0
    assert torch.allclose(mha(x)[:, :-1], mha(y)[:, :-1], atol=1e-6)


def test_mask_buffer_still_in_state_dict(cfg):
    # older checkpoints contain 'mask'; keeping the buffer lets them load
    assert any(k.endswith("mask") for k in MultiHeadAttention(cfg).state_dict())
