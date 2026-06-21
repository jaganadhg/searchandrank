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
  - knowledge-distillation
  - msmarco
  - lite
metrics:
  - mrr
  - ndcg
---

# LITE Re-ranker (MS MARCO, DistilBERT) — independent reproduction

A faithful, from-scratch reproduction of **LITE (Learnable Late InTEraction)** from
*Efficient Document Ranking with Learnable Late Interactions* ([arXiv:2406.17968](https://arxiv.org/abs/2406.17968)).
LITE replaces ColBERT's fixed MaxSim operator with a small **learnable** scorer over the
query–document token-similarity matrix.

> **This is an independent research reproduction**, trained on a subset under a free-tier
> compute budget. It is **not** the authors' model and does **not** reach paper-scale
> numbers. See *Limitations*.

## Model

- **Encoder:** shared `distilbert-base-uncased` dual-encoder → token embeddings.
- **Scorer (Separable LITE):** similarity matrix `S = Q·Dᵀ` (query len 30 × doc len 200),
  a row-wise MLP over the doc axis (hidden 2400), a column-wise MLP over the query axis
  (hidden 360), then a linear projection of the flattened matrix to a scalar.
- **Baseline for comparison:** a ColBERT-style MaxSim scorer (same encoder, no learned MLPs).

## Training

- **Objective:** Margin-MSE knowledge distillation from the cross-encoder teacher
  `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Data:** MS MARCO v2.1 `train`, ~61k (query, positive, negative) triplets.
- **Schedule:** batch 64, 20,000 steps, AdamW lr 2.8e-5, mixed precision, no early stopping.
- **Compute:** Kaggle free T4×2, ~3.7 h.

## Results (held-out MS MARCO `dev`/validation)

Reranking each query's candidate passages (same candidates for both models):

| Model | MRR@10 | nDCG@10 |
|---|---|---|
| **LITE (this model)** | **0.704** | **0.775** |
| MaxSim baseline | 0.612 | 0.705 |

LITE beats the MaxSim baseline by **+0.09 MRR@10 (+15%)**, a margin that is stable across
500/1000/2000-query evaluation slices — reproducing the paper's central claim that the
learnable interaction outperforms fixed MaxSim. A companion **Small-LITE** projection
(d′ 768→128) shrinks the cached document embeddings **~5.9×** (36.9 MB → 6.2 MB).

## Intended use

Research / educational re-ranking: given a query and a candidate set of passages, score and
re-order the candidates. Best used as a second-stage re-ranker over a first-stage retriever.

## How to use

The architecture is custom, so this repo ships the `literank` package alongside the weights.

```bash
git clone https://huggingface.co/jaganadhg/literank-msmarco-distilbert
cd literank-msmarco-distilbert
pip install torch transformers   # plus: datasets scikit-learn (for training/eval)
python load_example.py
```

```python
import torch
from literank.config import ModelConfig
from literank.model import Ranker
from literank.checkpoint import load_checkpoint

ckpt = torch.load("model.pt", map_location="cpu", weights_only=False)
ranker = Ranker(ModelConfig(**ckpt["config"]))
load_checkpoint("model.pt", ranker)            # loads weights (optimizer state omitted)
ranker.eval()

query = "what is late interaction in retrieval?"
docs = [
    "LITE is a learnable late-interaction re-ranker for document retrieval.",
    "Bananas are a good source of potassium.",
]
with torch.no_grad():
    scores = ranker.score([query] * len(docs), docs)
print(scores)   # higher = more relevant
```

## Limitations & honest caveats

- **Absolute scores (~0.70 MRR@10) are NOT comparable to the paper's 0.393.** The eval
  reranks each query's ~10 own passages (an easier pool) rather than BM25 top-1000. Only
  the **relative** LITE-vs-MaxSim comparison is claimed.
- **Subset / fixed budget**, not paper scale (the paper uses batch 128 / ~1.5M steps).
- The `relu` activation and KL/Margin-MSE weighting are inferred from the paper, not read
  verbatim from authors' code.
- Trained and evaluated only on MS MARCO (English passage ranking); zero-shot/BEIR behavior
  is untested here.

## Citation

```bibtex
@article{ji2024lite,
  title  = {Efficient Document Ranking with Learnable Late Interactions},
  author = {Ji, Ziwei and others},
  journal= {arXiv preprint arXiv:2406.17968},
  year   = {2024}
}
```

Reproduction code, tests, training notebook, and full results: see the project repository.
This model card documents an independent reproduction; credit for the LITE method belongs
to the original authors.
