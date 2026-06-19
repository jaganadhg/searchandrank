import torch
import torch.nn as nn
from literank.config import ModelConfig, TrainConfig
from literank.losses import distill_loss
from literank.train import train


class ToyRanker(nn.Module):
    """Text-driven, learnable, no tokenizer — for loop/checkpoint tests."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))

    def score(self, queries, docs):
        base = torch.tensor([float(len(d)) for d in docs])
        return self.w * base


def _triplets():
    # pos text longer than neg; teacher prefers pos
    return [{"query": "q", "pos": "aaaaaa", "neg": "aa", "t_pos": 5.0, "t_neg": 0.0}] * 8


def _toy_loss(model, batch):
    with torch.no_grad():
        s_pos = model.score(batch["query"], batch["pos"])
        s_neg = model.score(batch["query"], batch["neg"])
        return distill_loss(s_pos, s_neg, batch["t_pos"], batch["t_neg"]).item()


def test_training_decreases_loss(tmp_path):
    tcfg = TrainConfig(batch_size=4, max_steps=30, amp=False, seed=0,
                       checkpoint_every=1000, checkpoint_dir=str(tmp_path))
    triplets = _triplets()
    from literank.data import collate_triplets
    fixed_batch = collate_triplets(triplets[:4])

    model = ToyRanker()
    loss_before = _toy_loss(model, fixed_batch)

    train(ModelConfig(), tcfg, triplets, model=model, device="cpu")
    loss_after = _toy_loss(model, fixed_batch)

    # weight should have moved off zero toward positive margin
    assert model.w.item() != 0.0
    # the namesake assertion: training must actually reduce the loss
    assert loss_after < loss_before


def test_checkpoint_and_resume(tmp_path):
    tcfg = TrainConfig(batch_size=4, max_steps=5, amp=False, seed=0,
                       checkpoint_every=5, checkpoint_dir=str(tmp_path))
    train(ModelConfig(), tcfg, _triplets(), model=ToyRanker(), device="cpu")
    ckpt = tmp_path / "ckpt_step5.pt"
    assert ckpt.exists()
    # resume continues without error and runs more steps
    tcfg2 = TrainConfig(batch_size=4, max_steps=10, amp=False, seed=0,
                        checkpoint_every=5, checkpoint_dir=str(tmp_path))
    train(ModelConfig(), tcfg2, _triplets(), model=ToyRanker(),
          resume=str(ckpt), device="cpu")
    assert (tmp_path / "ckpt_step10.pt").exists()
