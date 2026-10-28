import yaml
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    dropout: float
    qkv_bias: bool
    tie_weights: bool = False  # share the input embedding with the output head (saves vocab_size * emb_dim params)

    def __post_init__(self):
        assert self.emb_dim % self.n_heads == 0, (
            f"emb_dim ({self.emb_dim}) must be divisible by n_heads ({self.n_heads})"
        )


def load_config(path: str) -> GPTConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return GPTConfig(**data)
