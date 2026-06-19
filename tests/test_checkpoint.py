import torch
import torch.nn as nn
from literank.checkpoint import save_checkpoint, load_checkpoint


def test_save_then_load_restores_state_and_step(tmp_path):
    model = nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(3, 4)).sum().backward()
    opt.step()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(str(path), model, opt, None, step=123, config={"a": 1})

    model2 = nn.Linear(4, 2)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    step = load_checkpoint(str(path), model2, opt2)
    assert step == 123
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)
