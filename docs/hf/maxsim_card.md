---
license: apache-2.0
language:
  - en
library_name: pytorch
pipeline_tag: text-ranking
base_model: distilbert-base-uncased
datasets:
  - microsoft/ms_marco
tags:
  - information-retrieval
  - passage-ranking
  - re-ranking
  - late-interaction
  - colbert
  - maxsim
  - knowledge-distillation
  - msmarco
  - baseline
metrics:
  - mrr
  - ndcg
---

# MaxSim Re-ranker (MS MARCO, DistilBERT) — baseline for LITE reproduction

A ColBERT-style **MaxSim** late-interaction re-ranker, trained as the **baseline** for an
independent reproduction of **LITE** ([arXiv:2406.17968](https://arxiv.org/abs/2406.17968)).
MaxSim scores a query–document pair as `Σ_i max_j (q_i · d_j)` over token embeddings — the
**fixed** operator that LITE replaces with a learnable scorer.

> This model exists to quantify what the learnable interaction adds. The learnable
> counterpart is **[jaganadhg/literank-msmarco-distilbert](https://huggingface.co/jaganadhg/literank-msmarco-distilbert)**.

## Model

- **Encoder:** shared `distilbert-base-uncased` dual-encoder → token embeddings.
- **Scorer (MaxSim):** `S = Q·Dᵀ`, then sum over query tokens of the max over (valid) doc
  tokens. **No learned parameters beyond the encoder.**

## Training

Identical recipe to the LITE model, for a fair comparison:

- **Objective:** Margin-MSE distillation from `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Data:** MS MARCO v2.1 `train`, ~300k (query, positive, negative) triplets (subset of 500k rows).
- **Schedule:** batch 64, AdamW lr 2.8e-5, AMP, early stopping on held-out dev MRR.
  This checkpoint is the `best.pt`; it plateaued and stopped around step 14k.
- **Compute:** Kaggle free T4×2 (trained in parallel with the LITE model).

## Results (held-out MS MARCO `dev`/validation, 2000 queries)

| Model | MRR@10 | nDCG@10 |
|---|---|---|
| LITE (learnable) | 0.724 | 0.791 |
| **MaxSim (this model)** | **0.664** | **0.745** |

The learnable LITE scorer beats this MaxSim baseline by **+0.06 MRR@10 (+9%)**. (A smaller-data
run showed a wider +15% gap; with ~5x more data this baseline improved more, narrowing the
margin — a reminder that a reported advantage depends on how well-trained the baseline is.)

## How to use

```bash
git clone https://huggingface.co/jaganadhg/maxsim-msmarco-distilbert
cd maxsim-msmarco-distilbert && pip install torch transformers
python load_example.py
```

```python
import torch
from literank.config import ModelConfig
from literank.model import Ranker
from literank.checkpoint import load_checkpoint

ckpt = torch.load("model.pt", map_location="cpu", weights_only=False)
ranker = Ranker(ModelConfig(**ckpt["config"]))   # config has scorer="maxsim"
load_checkpoint("model.pt", ranker)
ranker.eval()
scores = ranker.score(["query"] * 2, ["a relevant passage", "an irrelevant one"])
print(scores)
```

## Limitations & honest caveats

- **Absolute scores are NOT comparable to the paper's 0.393** — eval reranks each query's
  ~10 own passages, not BM25 top-1000. Only the relative LITE-vs-MaxSim gap is claimed.
- Subset / fixed-budget training, not paper scale.
- This is a **baseline**, intentionally less expressive than LITE (no learnable output scale),
  which is why its distillation loss floors higher (~3.6 vs LITE's ~0.74).

## Citation

```bibtex
@article{ji2024lite,
  title  = {Efficient Document Ranking with Learnable Late Interactions},
  author = {Ji, Ziwei and others},
  journal= {arXiv preprint arXiv:2406.17968},
  year   = {2024}
}
```

## Source code

Full implementation, tests, Kaggle training notebook, and detailed results:
**https://github.com/jaganadhg/searchandrank** — branch **`feat/paper-faithful`**.

This HF repo already bundles the `literank/` package, so `python load_example.py` works
straight after cloning. To train/evaluate from scratch, use the GitHub branch above.

Independent reproduction; credit for the LITE method and the MaxSim/ColBERT formulation
belongs to their original authors.
