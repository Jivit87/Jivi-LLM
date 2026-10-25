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

        