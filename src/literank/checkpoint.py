import glob
import os
import re

import torch


def prune_checkpoints(checkpoint_dir, keep_last, pattern="ckpt_step*.pt"):
    """Keep only the newest `keep_last` ckpt_step*.pt files (by step number).
    No-op when keep_last is falsy/<=0. Never touches files outside the pattern
    (e.g. best.pt)."""
    if not keep_last or keep_last <= 0:
        return
    paths = glob.glob(os.path.join(checkpoint_dir, pattern))

    def step_of(p):
        m = re.search(r"ckpt_step(\d+)\.pt$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    for old in sorted(paths, key=step_of)[:-keep_last]:
        try:
            os.remove(old)
        except OSError:
            pass


def save_checkpoint(path, model, optimizer, scaler, step, config):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "config": config,
    }, path)


def load_checkpoint(path, model, optimizer=None, scaler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt["step"]
