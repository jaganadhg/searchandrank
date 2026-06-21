import os
from literank.checkpoint import prune_checkpoints


def _touch(d, name):
    p = os.path.join(d, name)
    open(p, "w").close()
    return p


def test_prune_keeps_newest_by_step(tmp_path):
    d = str(tmp_path)
    for s in (1000, 2000, 3000, 4000, 5000):
        _touch(d, f"ckpt_step{s}.pt")
    best = _touch(d, "best.pt")
    prune_checkpoints(d, keep_last=2)
    remaining = sorted(os.listdir(d))
    assert remaining == ["best.pt", "ckpt_step4000.pt", "ckpt_step5000.pt"]
    assert os.path.exists(best)  # non-matching files untouched


def test_prune_noop_when_keep_last_zero(tmp_path):
    d = str(tmp_path)
    for s in (1000, 2000, 3000):
        _touch(d, f"ckpt_step{s}.pt")
    prune_checkpoints(d, keep_last=0)
    assert len(os.listdir(d)) == 3
