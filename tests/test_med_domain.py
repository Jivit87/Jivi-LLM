import os

import pytest
import torch

from jivi_llm.config import GPTConfig, load_config
from jivi_llm.model import GPTModel
from jivi_llm.tokenizer import get_tokenizer, encode, decode

MED_TOKENIZER = "data/medical/tokenizer/med_bpe_32k.json"


def tiny(tie):
    return GPTConfig(vocab_size=100, context_length=16, emb_dim=32, n_heads=4, n_layers=2, dropout=0.0,
                     qkv_bias=False, tie_weights=tie)


def test_tie_weights_shares_one_parameter():
    m = GPTModel(tiny(True))
    assert m.lm_head.weight is m.emb.tok_emb.weight


def test_tie_weights_saves_vocab_times_dim_params():
    untied = sum(p.numel() for p in GPTModel(tiny(False)).parameters())
    tied = sum(p.numel() for p in GPTModel(tiny(True)).parameters())
    assert untied - tied == 100 * 32


def test_tied_model_forward_shape():
    m = GPTModel(tiny(True))
    assert m(torch.randint(0, 100, (2, 8))).shape == (2, 8, 100)


def test_old_configs_default_to_untied():
    cfg = GPTConfig(vocab_size=100, context_length=16, emb_dim=32, n_heads=4, n_layers=2, dropout=0.0, qkv_bias=False)
    assert cfg.tie_weights is False


def test_med_config_is_about_124m_params():
    cfg = load_config("configs/med_124m.yaml")
    with torch.device("meta"):
        m = GPTModel(cfg)
    n = sum(p.numel() for p in m.parameters())
    assert 123e6 < n < 126e6, n


@pytest.mark.skipif(not os.path.exists(MED_TOKENIZER), reason="custom tokenizer not built")
def test_custom_tokenizer_round_trip_and_medical_terms():
    tok = get_tokenizer(MED_TOKENIZER)
    text = "Patients with thrombocytopenia and glomerulonephritis were treated with methotrexate."
    assert decode(encode(text, tok), tok) == text
    assert tok.n_vocab == 32000 and tok.eot_token == 0
    assert len(encode(" thrombocytopenia", tok)) == 1
