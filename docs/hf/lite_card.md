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
- **Data:** MS MARCO v2.1 `train`, ~300k (query, positive, negative) triplets (subset of 500k rows).
- **Schedule:** batch 64, AdamW lr 2.8e-5, mixed precision, early stopping on held-out dev MRR.
  This checkpoint is the `best.pt`, reached around step 44k.
- **Compute:** Kaggle free T4×2, multi-session with checkpoint/resume.

## Results (held-out MS MARCO `dev`/validation, 2000 queries)

Reranking each query's candidate passages (same candidates for both models):

| Model | MRR@10 | nDCG@10 |
|---|---|---|
| **LITE (this model)** | **0.724** | **0.791** |
| MaxSim baseline | 0.664 | 0.745 |

LITE beats the MaxSim baseline by **+0.06 MRR@10 (+9%)**, reproducing the paper's central
claim that the learnable interaction outperforms fixed MaxSim. (An earlier smaller-data run
showed a wider +15% gap; training both models on ~5x more data lifted the baseline and
narrowed the margin, so the size of LITE's advantage is data- and budget-dependent.) A
companion **Small-LITE** projection (d′ 768→128) shrinks the cached document embeddings
**~5.9×** (36.9 MB → 6.2 MB).

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

## Source code

Full implementation, tests, Kaggle training notebook, and detailed results:
**https://github.com/jaganadhg/searchandrank** — branch **`feat/paper-faithful`**.

This HF repo already bundles the `literank/` package, so `python load_example.py` works
straight after cloning. To train/evaluate from scratch, use the GitHub branch above.

This model card documents an independent reproduction; credit for the LITE method belongs
to the original authors.
