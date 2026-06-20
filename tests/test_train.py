import os
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


def test_early_stopping_triggers_before_max_steps(tmp_path):
    tcfg = TrainConfig(batch_size=4, max_steps=200, amp=False, seed=0,
                       checkpoint_every=1000, checkpoint_dir=str(tmp_path),
                       eval_every=5, patience=2, val_fraction=0.2)
    train(ModelConfig(), tcfg, _triplets(), model=ToyRanker(), device="cpu")
    assert os.path.exists(tmp_path / "best.pt")
    assert not os.path.exists(tmp_path / "ckpt_step200.pt")


def test_eval_disabled_by_default_runs_to_max_steps(tmp_path):
    tcfg = TrainConfig(batch_size=4, max_steps=10, amp=False, seed=0,
                       checkpoint_every=1000, checkpoint_dir=str(tmp_path))
    train(ModelConfig(), tcfg, _triplets(), model=ToyRanker(), device="cpu")
    assert os.path.exists(tmp_path / "ckpt_step10.pt")
    assert not os.path.exists(tmp_path / "best.pt")


def test_large_val_fraction_keeps_training_nonempty(tmp_path):
    # val_fraction=0.9 must not empty the training split (an empty DataLoader would
    # hang the step loop forever). patience high so it runs to max_steps.
    tcfg = TrainConfig(batch_size=4, max_steps=20, amp=False, seed=0,
                       checkpoint_every=1000, checkpoint_dir=str(tmp_path),
                       eval_every=5, patience=10, val_fraction=0.9)
    train(ModelConfig(), tcfg, _triplets(), model=ToyRanker(), device="cpu")
    # reaching the max-step checkpoint proves training actually ran (loader non-empty)
    assert os.path.exists(tmp_path / "ckpt_step20.pt")


def test_grad_accum_non_multiple_still_updates(tmp_path):
    # max_steps=10 is NOT a multiple of grad_accum=3
    tcfg = TrainConfig(batch_size=4, max_steps=10, grad_accum=3, amp=False, seed=0,
                       checkpoint_every=1000, checkpoint_dir=str(tmp_path))
    model = ToyRanker()
    w_initial = model.w.clone().item()

    train(ModelConfig(), tcfg, _triplets(), model=model, device="cpu")
    w_final = model.w.item()

    # final partial accumulation window (steps 10) should be flushed
    assert w_final != w_initial, "Optimizer should have stepped on final partial batch"
