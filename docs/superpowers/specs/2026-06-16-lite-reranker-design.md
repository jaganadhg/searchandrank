# LITE Re-ranker — Faithful Reference Implementation Design

**Date:** 2026-06-16
**Paper:** *Efficient Document Ranking with Learnable Late Interactions* (LITE) — https://arxiv.org/abs/2406.17968
**Status:** Approved design, ready for implementation planning

## Goal

A **faithful, architecturally-exact** reference implementation of LITE (Learnable late
InTEraction) for document ranking. Correctness and clarity over leaderboard numbers.
It must train and validate end-to-end on **small MS MARCO subsets** on a local 6 GB
GTX 1060, and scale unchanged (via config) to larger GPUs to approach the paper's
results. This is **not** a full paper reproduction (which needs cloud-scale compute:
BERT, batch 128, ~1.5M steps) and **not** a throwaway prototype.

## Non-goals

- First-stage ANN retrieval / ColBERT-PLAID-style token index (re-ranking only).
- A non-separable LITE scorer variant (the scorer is made pluggable, but only the
  Separable variant is built now — YAGNI).
- Matching the paper's exact MRR@10 numbers locally.

## Hardware / environment constraints

- Local GPU: NVIDIA GTX 1060 Max-Q, 6 GB VRAM, Pascal (sm_61), CUDA driver 12.2.
- Environment is already established with **uv** at the repo root:
  - Python pinned to 3.12; `requires-python = ">=3.12,<3.13"`.
  - `torch==2.5.1+cu121` (matches the 12.2 driver), via a configured `pytorch-cu121`
    uv index/source. **Do not** let resolution drift to cu130/cu128 wheels — they
    require a newer driver and fall back to CPU.
  - Deps: torch, transformers, datasets, numpy, scikit-learn, accelerate, tqdm.
  - Dev group: pytest, ruff.
  - The old Poetry draft under `./literank/` is kept only as reference; the new
    package lives under `src/literank/`.

## Background: what the paper does

- **Dual-encoder (DE)**, shared weights, pretrained BERT (paper: 6 layers, 768-dim),
  produces **token-level** embeddings for query and document independently.
- **Similarity matrix** `S = Q · Dᵀ`, where `Q ∈ ℝ^{L₁×d}` (query tokens) and
  `D ∈ ℝ^{L₂×d}` (doc tokens). Default `L₁=30`, `L₂=200`, `d=768`.
- **Separable LITE scorer** (paper Eq. 3–4): two-layer MLPs applied over the
  **sequence-length** axes of `S` (not the embedding axis), on **fixed-length padded**
  matrices, followed by a linear projection of the flattened result to a scalar.
- **Training:** distillation from a cross-encoder teacher (Hofstätter T2 in the paper)
  using **Margin-MSE** (+ KL) over `(query, positive, negative)` triplets. AdamW,
  batch 128, peak lr 2.8e-5, ~1.5M steps (paper scale).
- **Results (paper):** MS MARCO dev MRR@10 0.393 (> ColBERT 0.383); BEIR zero-shot
  wins on 11/14; "Small LITE" reaches ~0.25× storage via embedding-dim projection.

### Key correction vs. the existing draft

The draft in `./literank/src/literank.py` (a) uses BERT `[CLS]`-only vectors (one
vector per text), so there is no real token-level interaction, and (b) applies the
MLPs over the 768 embedding dimension of the similarity matrix. Both are wrong. LITE
requires **per-token** embeddings and MLPs over the **sequence-length** axes. This
reference implementation fixes both.

## Locked design decisions

1. **Encoder:** configurable HF model, default `distilbert-base-uncased`
   (6 layers / 768-dim — closest match to the paper, light on 6 GB).
2. **Encoder is fine-tuned jointly** with the scorer (paper-faithful). Config flag
   `freeze_encoder` exists only as a clearly-labeled non-faithful fast path for local
   smoke tests.
3. **Scorer:** Separable LITE only, built as a pluggable module.
4. **Activation σ:** ReLU (default, configurable). *Inferred detail* — the paper
   writes a generic σ; revisit against the authors' reference code if it surfaces.
5. **Projection / "Small LITE" lever:** an optional `Linear(d → d')` applied to token
   embeddings before similarity. Default `d'=768` (no reduction = Separable LITE);
   `d' < 768` yields Small LITE and shrinks the cache. Included now because it is the
   paper's headline storage/latency result.
6. **Loss:** Margin-MSE primary (on by default); KL secondary with weight `λ`
   (default `λ=0`, available). *The KL/Margin-MSE weighting is an inferred detail —
   verify against authors' reference if available.*
7. **Pipeline:** offline embedding cache + late-interaction **re-ranking** + eval
   (MRR@10, nDCG@10). No first-stage retrieval.
8. **Training-time encoding:** on-the-fly (encoder is trainable). The disk cache is
   used for the rerank/eval path.

## Architecture

### Component: `DualEncoder` (`encoder.py`)

- Wraps a shared HF encoder (`AutoModel`, default `distilbert-base-uncased`) and
  tokenizer.
- `encode(texts, max_len) -> (embeddings, mask)`:
  - Tokenize/pad/truncate to fixed `max_len` (`L₁` for queries, `L₂` for docs).
  - Forward pass → last hidden state `[B, L, d]`.
  - **Zero out padded positions** using the attention mask (no leakage into `S`).
  - Optional **L2-normalization** of token embeddings (configurable, default on —
    makes `S` a cosine-similarity matrix and stabilizes training).
  - Optional **projection** `Linear(d → d')` (the Small LITE lever).
- Used trainable in training; can be frozen and reused to populate the cache.
- *Unit answers:* produces fixed-shape, mask-clean token embeddings for a batch of
  texts; depends only on a HF model name + config.

### Component: `LITEScorer` (`model.py`)

Input: query embeddings `Q [B, L₁, d']`, doc embeddings `D [B, L₂, d']`.

1. **Similarity:** `S = einsum("bid,bjd->bij", Q, D)` → `[B, L₁, L₂]`.
2. **Row-wise MLP** over the doc axis (`L₂`), applied to each row:
   `S' = LN(σ(W₂ · LN(σ(W₁·S + b₁)) + b₂))`, hidden `m₂=2400` (`L₂→m₂→L₂`).
3. **Column-wise MLP** over the query axis (`L₁`), applied to each column:
   same form with hidden `m₁=360` (`L₁→m₁→L₁`). Implemented by transposing the last
   two axes, applying the MLP over `L₁`, transposing back.
4. **Final score:** `Linear(L₁·L₂ → 1)` on `vec(S'')` → scalar per pair.

- LayerNorm placement follows the paper equation exactly:
  `x → W₁ → +b₁ → σ → LN → W₂ → +b₂ → σ → LN`.
- Padded rows/cols are structured zeros (from the encoder mask); the fixed-length
  MLPs see them consistently.
- *Unit answers:* maps a pair of token-embedding sets to a scalar relevance score;
  depends only on shape config `(L₁, L₂, d', m₁, m₂, σ)`.

### Component: `Teacher` (`teacher.py`)

- Wraps `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Scores `(query, passage)` pairs; supports **precomputing & caching** teacher scores
  for the training subset (avoids re-running the teacher each epoch).

### Component: data (`data.py`)

- MS MARCO triplet dataset for distillation: yields `(query, pos_passage, neg_passage)`
  plus cached teacher scores `(t⁺, t⁻)`.
- Negatives: in-subset BM25/random negatives from MS MARCO passages (documented
  choice; full hard-negative mining is out of local scope but config-extensible).
- Fixed-length collation to `L₁ × L₂`.

### Component: losses (`losses.py`)

- `margin_mse(s_pos, s_neg, t_pos, t_neg)` = `MSE((s⁺−s⁻), (t⁺−t⁻))`.
- `kl_distill(...)` secondary term, combined as `loss = margin_mse + λ·kl`.

### Component: training (`train.py`)

- AdamW (default lr 2.8e-5), optional `accelerate`.
- Per step: encode `(q, d⁺, d⁻)` on the fly → scorer → `s⁺, s⁻` → loss vs cached
  teacher scores → backward → step.
- Local defaults: small subset, batch 8–16, few epochs; checkpoint to disk.

### Component: offline encode (`encode_cache.py`)

- Encode a document collection → padded token embeddings (+ mask) → disk
  (`.npy`/`.pt`, with `d'` controlling size). Demonstrates the storage property and
  the Small LITE lever.

### Component: rerank (`rerank.py`)

- For each dev query: load candidate docs' cached embeddings → encode query →
  `LITEScorer` → sort candidates by score → ranked list.

### Component: evaluate (`evaluate.py`)

- MRR@10 and nDCG@10 over a dev subset (scikit-learn `ndcg_score`; MRR@10 computed
  directly).

### Component: CLI (`cli.py`)

- Subcommands: `train | encode | rerank | eval`, each taking a config.
- `config.py` holds `ModelConfig`, `DataConfig`, `TrainConfig` dataclasses with all
  paper hyperparameters (`L₁=30, L₂=200, m₁=360, m₂=2400, d=768, d'=768`) as defaults.

## Data flow

```
TRAIN:   triplets (q, d+, d-) ──encode(on-the-fly)──▶ Q, D+, D-
                                   │
         teacher scores (cached) ──┤──▶ LITEScorer ──▶ s+, s-
                                   ▼
                          loss = MarginMSE(s, t) + λ·KL ──▶ AdamW step

ENCODE:  docs ──DualEncoder──▶ padded token embs (+mask) ──▶ disk cache (d')

RERANK:  query ──encode──▶ Q ;  candidates ──load cache──▶ D_i
                          ──▶ LITEScorer ──▶ scores ──▶ sort

EVAL:    ranked lists ──▶ MRR@10, nDCG@10  (dev subset)
```

## Testing strategy (TDD)

Unit tests drive the build, written before implementation:

- **`model.py`:** scorer output shape `[B]`; gradient flows to all params; masking
  (padded positions can't change a valid pair's score given zeroed embeddings);
  separability (row then column) applied in the correct order.
- **Universal-approximation sanity:** on toy `S` matrices, the scorer can be trained
  to fit a known target (e.g., MaxSim) to low error — a lightweight echo of the
  paper's theorem.
- **`losses.py`:** Margin-MSE and KL match hand-computed values; `λ=0` disables KL.
- **`encoder.py`:** fixed output shapes; padded positions are exactly zero; projection
  changes `d→d'`.
- **`data.py`:** collation produces `L₁×L₂` fixed shapes; triplet structure correct.
- **Smoke test:** one full CPU train step on a 2-triplet toy batch runs and decreases
  loss.

## Risks & open items

- **Activation (σ) and KL/Margin-MSE weighting** are inferred from the paper, not read
  verbatim — marked for verification against the authors' reference code if it appears.
- **6 GB VRAM:** fine-tuning DistilBERT with 3 encodes/triplet is tight; mitigate with
  small batch, gradient accumulation, and the `freeze_encoder` smoke-test path. Numbers
  will not match the paper locally — that is expected and in-scope.
- **Negative sampling** locally is simplified vs. the paper's hard negatives;
  documented and config-extensible.

## Out of scope (future work)

First-stage ANN retrieval, non-separable LITE variant, hard-negative mining at scale,
full BEIR zero-shot suite, cloud-scale reproduction run.
