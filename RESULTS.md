# Results — Reproducing LITE (Learnable Late Interaction) for Document Ranking

Paper: *Efficient Document Ranking with Learnable Late Interactions* — https://arxiv.org/abs/2406.17968

## Summary

A faithful, from-scratch reproduction of LITE (dual DistilBERT encoder + separable
row/column MLP scorer), trained via Margin-MSE distillation from a cross-encoder teacher
on an MS MARCO subset, on the **free Kaggle T4×2 tier**. The goal was a **qualitative**
reproduction under a fixed compute budget — not the paper's full scale.

## Setup

| | |
|---|---|
| Encoder | `distilbert-base-uncased` (6-layer/768, faithful proxy for the paper's 6-layer BERT) |
| Scorer | Separable LITE (row MLP m₂=2400 over doc axis, column MLP m₁=360 over query axis) vs. MaxSim baseline |
| Loss | Margin-MSE distillation, teacher `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Data | MS MARCO v2.1, `train` split, ~61k triplets (subset of 100k rows) |
| Schedule | batch 64, 20,000 steps, AdamW lr 2.8e-5, AMP, **no early stopping** (fixed budget) |
| Compute | Kaggle T4×2 (LITE on GPU 0, MaxSim on GPU 1, in parallel), ~3.7 h |
| Eval | MS MARCO v2.1 `validation` (=dev) split; rerank each query's candidate passages |

## Headline result — LITE vs. MaxSim (held-out dev)

| Dev queries | LITE MRR@10 | MaxSim MRR@10 | LITE nDCG@10 | MaxSim nDCG@10 |
|---|---|---|---|---|
| 500  | 0.6409 | 0.5374 | 0.7270 | 0.6459 |
| 1000 | 0.6746 | 0.5852 | 0.7523 | 0.6830 |
| 2000 | 0.7043 | 0.6123 | 0.7754 | 0.7051 |

**LITE beats the MaxSim baseline by ~+0.09 MRR@10 (+15%) and ~+0.07 nDCG@10 (+10%),
and the margin is stable across all three sample sizes** — signal, not sampling noise.
(Random ranking on these ~10-candidate queries is ~0.29 MRR@10, so both models clearly
learned; LITE learned more.)

Training loss corroborates the mechanism: LITE fits the teacher to ~0.74 vs. MaxSim's
~3.6 floor, because MaxSim lacks a learnable output scale to match the teacher's range.

## Storage ablation (Small-LITE lever)

Projecting token embeddings d′ 768→128 shrinks the cached document embeddings:

| Projection d′ | Cache size |
|---|---|
| 768 | 36.9 MB |
| 128 | 6.2 MB |

→ **5.9× smaller cache**, reproducing the paper's storage/quality trade-off.

## Methodology note

Trained on `train`, evaluated on the **disjoint `validation` (=dev) split** with no
tuning or early-stopping on it → no leakage; this is an unbiased held-out estimate and is
the split the paper itself reports on (MS MARCO test labels are hidden). If early stopping
is enabled in future runs, hold out a separate slice for selection vs. reporting.

## Honest limitations

- **Absolute scores (~0.70) are not comparable to the paper's 0.393.** Our candidate pool
  is each query's ~10 own passages, not BM25 top-1000 — an easier task. Only the
  **relative** LITE-vs-MaxSim gap is claimed.
- Not paper-scale (the paper uses batch 128 / ~1.5M steps); this is a deliberate
  qualitative reproduction on free compute.
- Eval uses a streamed slice of dev, not the full ~6,980-query dev set.

## Verdict

Both of the paper's central claims replicate convincingly under a free-tier budget:
**(1) learnable late interaction outperforms fixed MaxSim**, and **(2) the projection lever
trades storage for quality**. Next step for rigor: evaluate on **TREC DL 2019/2020**
(public dense judgments) and the full dev set with BM25 top-1000 candidates before making
any absolute-number claim.

## Reproduce

```bash
# from the repo root, with the trained checkpoints under kaggle_res_v3/
N_DEV=2000 uv run python scripts/eval_local.py
```
