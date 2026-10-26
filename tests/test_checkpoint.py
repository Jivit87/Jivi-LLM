import pytest
import torch
import sys, os
sys.path.insert(0, "src")

from jivi_llm.config import load_config
from jivi_llm.model import GPTModel
from jivi_llm.checkpoint import save_checkpoints, load_checkpoints


@pytest.fixture
def setup(tmp_path):
    # fresh model + optimizer with one training step done
    cfg = load_config("configs/debug.yaml")
    model = GPTModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randint(0, cfg.vocab_size, (2, cfg.context_length))
    import torch.nn.functional as F
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1))
    loss.backward()
    optimizer.step()

    path = str(tmp_path / "ckpt.pt")
    return model, optimizer, cfg, x, path


def test_save_load_outputs_match(setup):
    # loaded model must produce identical outputs to saved model
    model, optimizer, cfg, x, path = setup
    save_checkpoints(path, model, optimizer, step=42, config=cfg)

    model2 = GPTModel(cfg)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    load_checkpoints(path, model2, optimizer2)

    model.eval(); model2.eval()
    with torch.no_grad():
        out1 = model(x)
        out2 = model2(x)
    assert torch.allclose(out1, out2), "outputs diverged after load"


def test_step_is_saved(setup):
    # metadata step must match what was passed to save
    model, optimizer, cfg, x, path = setup
    save_checkpoints(path, model, optimizer, step=99, config=cfg)

    model2 = GPTModel(cfg)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    meta = load_checkpoints(path, model2, optimizer2)
    assert meta["step"] == 99
