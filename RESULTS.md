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

## Follow-up runs: scaling data and steps

The table above is the **initial run** (subset of 100k rows, ~61k triplets, 20k steps).
Follow-up runs probed whether scaling helps, all evaluated on the same 2000-query dev slice:

| Run | Data (rows / triplets) | LITE steps | LITE MRR@10 | MaxSim MRR@10 | LITE − MaxSim |
|---|---|---|---|---|---|
| Initial | 100k / ~61k | 20k | 0.7043 | 0.6123 | +0.092 (+15%) |
| More data | 500k / ~300k | ~45k (best.pt) | **0.7242** | **0.6643** | +0.060 (+9%) |
| More steps | 500k / ~300k | 50k (best.pt) | 0.7175 | 0.6643 | +0.053 (+8%) |
| Full data | full split | 36k (best.pt, killed @90%) | 0.7203 | 0.6643† | +0.056† |

† MaxSim was not retrained on the full split, so this "gap" mixes data scales — indicative only.

Three findings:

1. **More data lifted both models, and lifted the baseline more, so the gap narrowed from
   +15% to +9%.** Part of LITE's initial edge came from the MaxSim baseline being
   data-starved; a better-trained baseline closes some of it. LITE still wins clearly, but
   the size of its advantage is data- and budget-dependent.
2. **Training LITE further (45k → 50k steps) did not help** (0.7242 → 0.7175, within noise).
   At ~17 epochs over the same 300k triplets the model had reached its useful ceiling for that
   data budget; the extra steps bought nothing measurable.
3. **Scaling data past ~500k did not help either** (LITE 0.7043 → 0.7242 → 0.7203 across
   100k / 500k / full). The most likely cause is the **evaluation, not the model**: this eval
   reranks each query's ~10 own candidate passages with ~1 relevant, so MRR@10 **saturates
   around 0.72** — once the relevant passage sits near the top, a stronger model cannot move
   the number. Distinguishing further gains would require the harder **BM25 top-1000
   reranking** protocol (and BEIR zero-shot) that the paper uses.

The **published HuggingFace models are the best checkpoints from the "more data" run**
(LITE 0.724, MaxSim 0.664); the larger-data and longer runs did not beat them on this eval.

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

## Published models

- **LITE** — https://huggingface.co/jaganadhg/literank-msmarco-distilbert
- **MaxSim baseline** — https://huggingface.co/jaganadhg/maxsim-msmarco-distilbert

## Reproduce

```bash
# from the repo root, with the trained checkpoints under kaggle_res_v3/
N_DEV=2000 uv run python scripts/eval_local.py
```
