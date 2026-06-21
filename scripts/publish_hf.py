"""Publish a trained literank checkpoint to the HuggingFace Hub.

Builds a self-contained model repo: slimmed weights (optimizer state dropped),
config.json, the bundled `literank` package, a load example, and the model card.

Usage (from repo root):
    uv run python scripts/publish_hf.py \
        --ckpt kaggle_res_v3/ckpt_maxsim/ckpt_step20000.pt \
        --repo-id jaganadhg/maxsim-msmarco-distilbert \
        --card docs/hf/maxsim_card.md
"""
import argparse
import json
import os
import shutil
import tempfile

import torch
from huggingface_hub import create_repo, upload_folder

LOAD_EXAMPLE = '''\
"""Minimal load + score example. Run from this repo dir: python load_example.py"""
import torch
from literank.config import ModelConfig
from literank.model import Ranker
from literank.checkpoint import load_checkpoint

ckpt = torch.load("model.pt", map_location="cpu", weights_only=False)
ranker = Ranker(ModelConfig(**ckpt["config"]))
load_checkpoint("model.pt", ranker)
ranker.eval()

query = "what is late interaction in retrieval?"
docs = [
    "LITE is a learnable late-interaction re-ranker for document retrieval.",
    "Bananas are a good source of potassium.",
]
with torch.no_grad():
    scores = ranker.score([query] * len(docs), docs)
for s, d in sorted(zip(scores.tolist(), docs), reverse=True):
    print(f"{s:8.3f}  {d}")
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to a training checkpoint (.pt)")
    ap.add_argument("--repo-id", required=True, help="e.g. jaganadhg/maxsim-msmarco-distilbert")
    ap.add_argument("--card", required=True, help="path to the model card README.md")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--package", default="src/literank")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    with tempfile.TemporaryDirectory() as d:
        torch.save({"model": ckpt["model"], "config": ckpt["config"], "step": ckpt["step"]},
                   os.path.join(d, "model.pt"))
        json.dump(ckpt["config"], open(os.path.join(d, "config.json"), "w"), indent=2)
        shutil.copytree(args.package, os.path.join(d, "literank"),
                        ignore=shutil.ignore_patterns("__pycache__"))
        with open(os.path.join(d, "load_example.py"), "w") as f:
            f.write(LOAD_EXAMPLE)
        shutil.copy(args.card, os.path.join(d, "README.md"))

        create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        url = upload_folder(repo_id=args.repo_id, folder_path=d,
                            commit_message="Publish literank checkpoint: weights, code, model card")
        print("published:", url)


if __name__ == "__main__":
    main()
