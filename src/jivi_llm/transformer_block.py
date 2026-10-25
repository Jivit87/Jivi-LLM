import torch.nn as nn 
from jivi_llm.attention import MultiHeadAttention 
from jivi_llm.model import LayerNorm, FeedForward 

class TransformerBlock(nn.Module): 

    # x → LayerNorm → MultiHeadAttention → dropout → + x  (residual)
    # → LayerNorm → FeedForward        → dropout → + x  (residual)
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = LayerNorm(cfg.emb_dim)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = LayerNorm(cfg.emb_dim)
        self.ff = FeedForward(cfg) 
        self.drop = nn.Dropout(cfg.dropout) 

    def forward(self, x):
        # shortcut connection 1 : adding x to the output of the attention sublayer (pre-norm + multi-head attention + dropout) 
        x = x + self.drop(self.attn(self.ln1(x)))

        # shortcut connection 2 : adding x to the output of the feedforward sublayer (pre-norm + feedforward + dropout) 
        x = x + self.drop(self.ff(self.ln2(x)))

        return x





         