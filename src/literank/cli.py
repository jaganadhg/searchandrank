import argparse
import logging
from literank.config import ModelConfig, TrainConfig, DataConfig


def build_parser():
    p = argparse.ArgumentParser(prog="literank")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train a ranker via distillation")
    t.add_argument("--scorer", choices=["lite", "maxsim"], default="lite")
    t.add_argument("--proj-dim", type=int, default=ModelConfig().proj_dim)
    t.add_argument("--max-steps", type=int, default=TrainConfig().max_steps)
    t.add_argument("--batch-size", type=int, default=TrainConfig().batch_size)
    t.add_argument("--grad-accum", type=int, default=TrainConfig().grad_accum,
                   help="micro-batches per optimizer step; effective batch = batch-size * grad-accum")
    t.add_argument("--subset-size", type=int, default=DataConfig().subset_size)
    t.add_argument("--checkpoint-dir", default=TrainConfig().checkpoint_dir)
    t.add_argument("--keep-last", type=int, default=TrainConfig().keep_last,
                   help="max ckpt_step*.pt files to keep (older pruned); 0 = keep all")
    t.add_argument("--eval-every", type=int, default=TrainConfig().eval_every)
    t.add_argument("--patience", type=int, default=TrainConfig().patience)
    t.add_argument("--log-every", type=int, default=TrainConfig().log_every)
    t.add_argument("--resume", default=None)
    t.add_argument("--device", default="cuda")

    e = sub.add_parser("encode", help="encode + cache document token embeddings")
    e.add_argument("--proj-dim", type=int, default=ModelConfig().proj_dim)
    e.add_argument("--out", required=False, default="doc_cache.pt")
    e.add_argument("--device", default="cuda")

    r = sub.add_parser("rerank", help="rerank cached candidates for dev queries")
    r.add_argument("--cache", default="doc_cache.pt")
    r.add_argument("--checkpoint", default=None)
    r.add_argument("--device", default="cuda")

    v = sub.add_parser("eval", help="compute MRR@10 / nDCG@10")
    v.add_argument("--runs", default="runs.json")

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Dispatch is wired to module entrypoints; the Kaggle notebook drives the
    # full train->encode->rerank->eval flow. Heavy paths require a GPU + data.
    if args.command == "train":
        from literank.data import build_msmarco_triplets
        from literank.teacher import CrossEncoderTeacher, cache_teacher_scores
        from literank.train import train
        mcfg = ModelConfig(scorer=args.scorer, proj_dim=args.proj_dim)
        tcfg = TrainConfig(max_steps=args.max_steps, checkpoint_dir=args.checkpoint_dir,
                          batch_size=args.batch_size, grad_accum=args.grad_accum,
                          keep_last=args.keep_last,
                          eval_every=args.eval_every, patience=args.patience,
                          log_every=args.log_every)
        dcfg = DataConfig(subset_size=args.subset_size)
        triplets = build_msmarco_triplets(dcfg)
        teacher = CrossEncoderTeacher(dcfg.teacher_name, device=args.device,
                                      max_len=dcfg.teacher_max_len)
        triplets = cache_teacher_scores(triplets, teacher, "teacher_scores.json")
        train(mcfg, tcfg, triplets, resume=args.resume, device=args.device)
    else:
        raise SystemExit(f"command '{args.command}' is driven from the Kaggle notebook")


if __name__ == "__main__":
    main()
