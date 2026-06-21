"""Local LITE-vs-MaxSim eval from the downloaded Kaggle checkpoints.

Builds a small MS MARCO v2.1 validation slice (streaming, no full download),
reranks each query's candidate passages with each trained model, and reports
MRR@10 / nDCG@10. Run from the repo root:  PYTHONPATH=src uv run python scripts/eval_local.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from datasets import load_dataset
from literank.config import ModelConfig
from literank.model import Ranker
from literank.checkpoint import load_checkpoint
from literank.evaluate import mrr_at_k, ndcg_at_k

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_DEV = int(os.environ.get("N_DEV", "500"))
RES = os.path.join(os.path.dirname(__file__), "..", "kaggle_res_v3")


def build_dev(n):
    ds = load_dataset("microsoft/ms_marco", "v2.1", split="validation", streaming=True)
    dev = []
    for rec in ds:
        docs = rec["passages"]["passage_text"]
        labels = rec["passages"]["is_selected"]
        if sum(labels) == 0 or len(docs) < 2:
            continue
        dev.append({"query": rec["query"], "docs": docs, "labels": labels})
        if len(dev) >= n:
            break
    return dev


@torch.no_grad()
def evaluate(scorer, ckpt_path, dev):
    ranker = Ranker(ModelConfig(scorer=scorer, proj_dim=768)).to(DEVICE)
    load_checkpoint(ckpt_path, ranker, map_location=DEVICE)
    ranker.eval()
    ranked = []
    for ex in dev:
        scores = ranker.score([ex["query"]] * len(ex["docs"]), ex["docs"])
        order = torch.argsort(scores, descending=True).tolist()
        ranked.append([ex["labels"][i] for i in order])
    return mrr_at_k(ranked, 10), ndcg_at_k(ranked, 10)


def main():
    print(f"device={DEVICE}  building {N_DEV}-query dev slice ...")
    dev = build_dev(N_DEV)
    print(f"dev queries with >=1 relevant: {len(dev)}")
    runs = [
        ("LITE",   "lite",   f"{RES}/ckpt_lite/ckpt_step20000.pt"),
        ("MaxSim", "maxsim", f"{RES}/ckpt_maxsim/ckpt_step20000.pt"),
    ]
    print(f"{'model':8} {'MRR@10':>8} {'nDCG@10':>8}")
    for name, scorer, ckpt in runs:
        mrr, ndcg = evaluate(scorer, ckpt, dev)
        print(f"{name:8} {mrr:8.4f} {ndcg:8.4f}")


if __name__ == "__main__":
    main()
