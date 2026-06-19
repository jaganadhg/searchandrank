import os
import torch
from torch.utils.data import DataLoader
from literank.data import TripletDataset, collate_triplets
from literank.losses import distill_loss
from literank.checkpoint import save_checkpoint, load_checkpoint


def train_step(model, batch, optimizer, scaler, kl_weight, grad_accum, do_step, device):
    use_amp = scaler is not None and scaler.is_enabled()
    with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=use_amp):
        s_pos = model.score(batch["query"], batch["pos"])
        s_neg = model.score(batch["query"], batch["neg"])
        loss = distill_loss(s_pos, s_neg,
                            batch["t_pos"].to(s_pos.device),
                            batch["t_neg"].to(s_neg.device),
                            kl_weight=kl_weight)
    scaled = loss / grad_accum
    if use_amp:
        scaler.scale(scaled).backward()
    else:
        scaled.backward()
    if do_step:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    return float(loss.detach())


def train(model_cfg, train_cfg, triplets, model=None, resume=None, device="cpu"):
    torch.manual_seed(train_cfg.seed)
    if model is None:
        from literank.model import Ranker
        model = Ranker(model_cfg)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=train_cfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg.amp and device == "cuda")

    start_step = 0
    if resume:
        start_step = load_checkpoint(resume, model, optimizer, scaler, map_location=device)

    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    loader = DataLoader(TripletDataset(triplets), batch_size=train_cfg.batch_size,
                        shuffle=True, collate_fn=collate_triplets)
    model.train()
    optimizer.zero_grad()
    step = start_step
    while step < train_cfg.max_steps:
        for batch in loader:
            step += 1
            do_step = (step % train_cfg.grad_accum == 0) or step >= train_cfg.max_steps
            train_step(model, batch, optimizer, scaler, train_cfg.kl_weight,
                       train_cfg.grad_accum, do_step, device)
            if step % train_cfg.checkpoint_every == 0 or step >= train_cfg.max_steps:
                save_checkpoint(os.path.join(train_cfg.checkpoint_dir, f"ckpt_step{step}.pt"),
                                model, optimizer, scaler, step, vars(model_cfg))
            if step >= train_cfg.max_steps:
                break
    return model
