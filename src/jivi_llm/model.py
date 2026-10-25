import torch 
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    # map token id and positions to vectors 

    def __init__(self, cfg):
        super().__init__()
        # cfg.vocab_size : There are 50,257 possible token IDs.
        # cfg.emb_dim : Each token gets a vector of 768 numbers.
        # these are learnable parameters that the model will optimize during training 
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim) 
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):

        # x shape: (B, T): (row, col)
        # B: batch size
        # T: tokens per sequence
        B, T = x.shape 
        positions = torch.arange(T, device=x.device)
        out = self.tok_emb(x) + self.pos_emb(positions)
        return self.drop(out)


class LayerNorm(nn.Module):
    
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim)) # learnable scaling parameter 
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # learnable shift parameter

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * x_norm + self.shift   
        # we are scaling and adding shifts so that model learn during training and changes paramaters accordignly 


class FeedForward(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            nn.GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
            nn.Dropout(cfg.dropout) # helps to prevent overfitting 
        )

    def forward(self, x):
        return self.net(x)

class GPTModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        from jivi_llm.transformer_block import TransformerBlock
        self.emb = EmbeddingLayer(cfg)
        
        # stacking transformer blocks 
        self.blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )

        self.ln_final = LayerNorm(cfg.emb_dim)

        # output head: project from emb_dim to vocal logits
        self.lm_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, x):
        # x: (B, T)
        x = self.emb(x)       # (B, T, C)
        x = self.blocks(x)    # (B, T, C)
        x = self.ln_final(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits
            

    
