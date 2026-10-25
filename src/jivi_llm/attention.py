import torch
import torch.nn as nn
import math


class CausalSelfAttention(nn.Module):
    # single-head causal self-attention

    def __init__(self, cfg):
        super().__init__()
        self.d_k = cfg.emb_dim  # dimension of Q, K, V

        # linear projections for Q, K, V
        # nn.Linear(in_features, out_features, bias=False)
        self.W_q = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.qkv_bias)
        self.W_k = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.qkv_bias)
        self.W_v = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.qkv_bias)

        self.attn_drop = nn.Dropout(cfg.dropout)

        # causal mask: registers a (T, T) upper-triangular matrix inside the model.
        # True positions are future tokens the model must NOT look at.
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(cfg.context_length, cfg.context_length), diagonal=1).bool()
        )

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape

        Q = self.W_q(x)  # (B, T, C)
        K = self.W_k(x)
        V = self.W_v(x)

        # scaled dot-product scores — transpose last two dims for batched matmul
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (B, T, T)

        # mask out future positions — slice to actual T in case T < context_length
        scores = scores.masked_fill(self.mask[:T, :T], float("-inf"))

        weights = torch.softmax(scores, dim=-1)  # (B, T, T)
        weights = self.attn_drop(weights)

        out = weights @ V  # (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    # run n_heads attention in parallel 

    def __init__(self, cfg):
        super().__init__()
        assert cfg.emb_dim % cfg.n_heads == 0, "emb_dim must be divisible by n_heads"

        self.n_heads = cfg.n_heads
        self.d_head = cfg.emb_dim // self.n_heads # dimension per head 

        self.W_q = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.qkv_bias)    
        self.W_k = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.qkv_bias)
        self.W_v = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.qkv_bias)
        self.W_o = nn.Linear(cfg.emb_dim, cfg.emb_dim) # output projection 

        self.attn_drop = nn.Dropout(cfg.dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(cfg.context_length, cfg.context_length), diagonal=1).bool()
        )

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape 
        H = self.n_heads 
        d = self.d_head 

        Q = self.W_q(x)  # (B, T, C)
        K = self.W_k(x)
        V = self.W_v(x) 

        # split C into H heads: (B, T, C) -> (B, T, H, d) -> (B, H, T, d) 
        Q = Q.view(B, T, H, d).transpose(1, 2) 
        K = K.view(B, T, H, d).transpose(1, 2)  
        V = V.view(B, T, H, d).transpose(1, 2)  

        # scaled dot product  
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d)
        scores = scores.masked_fill(self.mask[:T, :T], float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        out = weights @ V  # (B, H, T, d) 

        # merge heads back: (B, H, T, d) -> (B, T, H, d) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)  # (B, T, C)



        