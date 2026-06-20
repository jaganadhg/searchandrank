import logging
import os
import torch
from torch.utils.data import DataLoader
from literank.data import TripletDataset, collate_triplets, build_eval_set
from literank.losses import distill_loss
from literank.checkpoint import save_checkpoint, load_checkpoint
from literank.evaluate import evaluate_ranker_mrr

logger = logging.getLogger("literank.train")


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

    early = bool(train_cfg.eval_every and train_cfg.eval_every > 0)
    if early:
        # Clamp so the held-out val split and the training split stay disjoint and
        # both non-empty (avoids a silent train==val leak for large val_fraction).
        n_val = max(1, min(int(len(triplets) * train_cfg.val_fraction), len(triplets) - 1))
        val_set = build_eval_set(triplets[:n_val])
        train_triplets = triplets[n_val:]
    else:
        val_set = None
        train_triplets = triplets

    loader = DataLoader(TripletDataset(train_triplets), batch_size=train_cfg.batch_size,
                        shuffle=True, collate_fn=collate_triplets)
    model.train()
    optimizer.zero_grad()
    step = start_step
    best_mrr = float("-inf")
    no_improve = 0
    stop = False
    best_path = os.path.join(train_cfg.checkpoint_dir, train_cfg.best_ckpt_name)
    loss_window = []

    logger.info(
        "start training: %d train / %d val triplets, batch %d x grad_accum %d "
        "(effective %d), device %s, max_steps %d, early_stopping %s",
        len(train_triplets), len(val_set) if val_set else 0,
        train_cfg.batch_size, train_cfg.grad_accum,
        train_cfg.batch_size * train_cfg.grad_accum, device,
        train_cfg.max_steps, "on" if early else "off",
    )
    while step < train_cfg.max_steps and not stop:
        for batch in loader:
            step += 1
            do_step = (step % train_cfg.grad_accum == 0) or step >= train_cfg.max_steps
            loss_window.append(train_step(model, batch, optimizer, scaler,
                                          train_cfg.kl_weight, train_cfg.grad_accum,
                                          do_step, device))
            if step % train_cfg.log_every == 0:
                logger.info("step %d/%d | loss %.4f", step, train_cfg.max_steps,
                            sum(loss_window) / len(loss_window))
                loss_window = []
            if early and step % train_cfg.eval_every == 0:
                mrr = evaluate_ranker_mrr(model, val_set, k=train_cfg.eval_k, device=device)
                if mrr > best_mrr + train_cfg.min_delta:
                    best_mrr = mrr
                    no_improve = 0
                    save_checkpoint(best_path, model, optimizer, scaler, step, vars(model_cfg))
                    logger.info("step %d | dev MRR@%d %.4f (new best, saved %s)",
                                step, train_cfg.eval_k, mrr, best_path)
                else:
                    no_improve += 1
                    logger.info("step %d | dev MRR@%d %.4f (best %.4f, no_improve %d/%d)",
                                step, train_cfg.eval_k, mrr, best_mrr,
                                no_improve, train_cfg.patience)
                    if no_improve >= train_cfg.patience:
                        stop = True
                        logger.info("early stopping at step %d (dev MRR@%d plateaued, best %.4f)",
                                    step, train_cfg.eval_k, best_mrr)
            if step % train_cfg.checkpoint_every == 0 or step >= train_cfg.max_steps:
                ckpt = os.path.join(train_cfg.checkpoint_dir, f"ckpt_step{step}.pt")
                save_checkpoint(ckpt, model, optimizer, scaler, step, vars(model_cfg))
                logger.info("step %d | saved checkpoint %s", step, ckpt)
            if step >= train_cfg.max_steps or stop:
                break
    logger.info("finished at step %d (%s)", step,
                "early-stopped" if stop else "reached max_steps")
    return model
