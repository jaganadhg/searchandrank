# literank — a LITE re-ranker reference implementation

`literank` (under `src/literank/`) is a from-scratch, faithful reference implementation of the
**LITE** late-interaction document re-ranker described in
[arXiv:2406.17968](https://arxiv.org/abs/2406.17968). It is written to be read alongside the
paper — separable row/column MLPs over the query/document token-similarity matrix, a learned
final projection, knowledge distillation from a cross-encoder teacher, and the "Small-LITE"
projection-dimension storage lever — and it is sized to **train and evaluate on Kaggle's free
GPU tier** (T4 x2), not on a multi-GPU cluster.

**Goal:** qualitative reproduction of the LITE architecture and training recipe, not a
leaderboard score. The paper reports **MRR@10 = 0.393** on the full MS MARCO dev set after
full-scale training. Running this code locally or on Kaggle, with a `--subset-size` slice of
MS MARCO and a handful of GPU-hours, **will score below 0.393** — that gap is expected. The
deliverable of this project is: the LITE scorer trains and ranks sensibly, the ablations behave
the way the paper predicts (LITE beats/matches MaxSim, smaller `proj_dim` shrinks the cache),
not a SOTA number.

## Pretrained models (HuggingFace)

The trained checkpoints from this branch are published as self-contained model repos
(slim weights + bundled `literank/` package + load example + model card):

- **LITE re-ranker** — https://huggingface.co/jaganadhg/literank-msmarco-distilbert (MRR@10 0.704)
- **MaxSim baseline** — https://huggingface.co/jaganadhg/maxsim-msmarco-distilbert (MRR@10 0.612)

(Numbers are on a ~10-candidate dev pool; see [RESULTS.md](RESULTS.md) for the honest caveat
that these are *relative* comparisons, not paper-comparable absolutes.)

## Package layout

```
src/literank/
  config.py        ModelConfig / TrainConfig / DataConfig (paper hyperparameters)
  encoder.py        DualEncoder: HF backbone (default distilbert-base-uncased) + mask + projection + L2-norm
  model.py          LITEScorer, MaxSimScorer, build_scorer(), Ranker (encoder + scorer)
  data.py           TripletDataset / collate_triplets + build_msmarco_triplets()
  teacher.py        CrossEncoderTeacher + (cache_)teacher_scores for distillation targets
  losses.py         margin_mse, kl_distill, distill_loss (Margin-MSE + optional KL, weight λ)
  train.py          train_step / train: AMP, grad accumulation, periodic checkpointing
  checkpoint.py     save_checkpoint / load_checkpoint
  encode_cache.py   encode_and_cache / save_embeddings / load_embeddings (offline doc cache)
  rerank.py         rerank(): score+sort cached candidate docs for one query
  evaluate.py       mrr_at_k, ndcg_at_k
  cli.py            python -m literank.cli {train|encode|rerank|eval}
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

`pyproject.toml` pins `torch` to the `pytorch-cu121` index, so `uv sync` on a CUDA-capable
machine pulls a CUDA build automatically; on Kaggle the preinstalled CUDA torch can be reused
instead (see the [Kaggle](#running-on-kaggle) section).

## Running tests

The test suite is split into fast offline unit tests and slower tests that need
network/model downloads (marked `integration` in `pyproject.toml`):

```bash
uv run pytest -m "not integration"   # fast, offline, safe for CI / smoke-testing a new env
uv run pytest                        # full suite, including integration tests
```

## CLI

The CLI entry point is `python -m literank.cli`, with one subcommand fully wired
end-to-end (`train`) and three subcommands (`encode`, `rerank`, `eval`) whose
underlying functions are intentionally driven directly from the Kaggle notebook
(`notebooks/literank_kaggle.ipynb`) rather than from the CLI, since they need a live GPU
session, a trained checkpoint, and dataset-specific glue that doesn't generalize well to a
fixed CLI contract.

```bash
# Train a Ranker (DualEncoder + scorer) via Margin-MSE distillation from a cross-encoder teacher.
python -m literank.cli train \
    --scorer {lite,maxsim} \      # default: lite
    --proj-dim 768 \              # projection dim d' (< embed_dim=768 => Small-LITE)
    --max-steps 20000 \
    --batch-size 16 \             # micro-batch size
    --grad-accum 1 \              # micro-batches per optimizer step (effective batch = batch-size * grad-accum)
    --subset-size 100000 \        # MS MARCO training rows to sample
    --checkpoint-dir checkpoints \
    --keep-last 3 \               # retain only the newest 3 ckpt_step*.pt (older pruned); 0 = keep all
    --log-every 50 \              # steps between progress (step/loss) log lines
    --eval-every 0 \              # >0 enables periodic dev-MRR@10 eval + early stopping
    --patience 3 \                # consecutive evals without improvement before stopping
    --resume <path/to/ckpt.pt> \  # optional: resume from a saved checkpoint
    --device cuda

# encode / rerank / eval are registered subcommands (see `literank.cli.build_parser`) but
# their dispatch raises SystemExit("command '<x>' is driven from the Kaggle notebook") --
# call the underlying functions directly, as the notebook does:
#   literank.encode_cache.encode_and_cache(encoder, docs, max_len, path)
#   literank.rerank.rerank(scorer, query_emb, query_mask, doc_embs, doc_masks)
#   literank.evaluate.mrr_at_k(...) / literank.evaluate.ndcg_at_k(...)
```

### Paper-faithful training

To match the paper's *recipe* (not just the subset defaults), use the full Separable-LITE
config, the paper's **effective batch of 128**, the paper's fixed-budget schedule, and **no
early stopping** (the paper trains to a fixed step count, ~1.5M steps, rather than stopping
on a dev metric):

```bash
python -m literank.cli train \
    --scorer lite --proj-dim 768 \   # full Separable LITE (not Small-LITE)
    --batch-size 16 --grad-accum 8 \ # effective batch 128 (16 x 8) to fit a 16 GB GPU
    --subset-size 1000000 \          # as much MS MARCO as you can afford
    --max-steps 1500000 \            # paper-scale fixed budget (no --eval-every)
    --checkpoint-dir ckpt_paper --device cuda
# lr defaults to 2.8e-5 (the paper's peak LR); encoder defaults to distilbert-base-uncased
# (a faithful 6-layer/768 proxy for the paper's 6-layer BERT).
```

**Honest caveat:** this is the faithful *configuration*, but the full ~1.5M-step / batch-128
run is **cloud-GPU scale** — it will not complete within free Kaggle's session/quota limits.
On free Kaggle, run this same config with a smaller `--max-steps`/`--subset-size`; you get
the qualitative reproduction (LITE > MaxSim, Small-LITE storage tradeoff), **not** the
paper's 0.393 MRR@10. Do **not** add `--eval-every` for a paper-faithful run — early stopping
is a compute guardrail we added, not part of the paper's method.

## Running on Kaggle

See `notebooks/literank_kaggle.ipynb` for the full, runnable flow. Summary:

1. Enable **GPU T4 x2** and **Internet** in the notebook sidebar.
2. `!pip install -q -e .` (or `!pip install torch transformers datasets scikit-learn` if not
   installing the package itself).
3. Smoke-test the environment offline: `!uv run pytest -m "not integration" -q`.
4. Train LITE: `python -m literank.cli train --scorer lite --proj-dim 768 --subset-size 100000
   --max-steps 20000 --checkpoint-dir /kaggle/working/ckpt_lite --device cuda`.
5. Train the MaxSim baseline the same way, with `--scorer maxsim` and a separate
   `--checkpoint-dir`.
6. Run the Small-LITE storage ablation (`--proj-dim 128` vs `768`, compare
   `encode_and_cache`'s returned byte count).
7. Encode dev passages, rerank them with the trained LITE checkpoint, and compute
   MRR@10 / nDCG@10 with `literank.evaluate`.

### Checkpoint -> Kaggle Dataset -> `--resume` loop

Kaggle GPU sessions are time-boxed, so reaching `--max-steps 20000` (or beyond) usually spans
multiple sessions:

1. At the end of a session, create a **Kaggle Dataset** from the `/kaggle/working/ckpt_lite`
   (or `ckpt_maxsim`) folder (Output pane -> "New Dataset"), which snapshots the
   `ckpt_step<N>.pt` checkpoint files written by `literank.checkpoint.save_checkpoint` during
   training.
2. In the next session, attach that Dataset as input data (mounted read-only under
   `/kaggle/input/<dataset-name>/`).
3. Copy the latest checkpoint to a writable path and resume. Because a session can be
   interrupted at any step, the checkpoint to resume from is whichever `ckpt_step<N>.pt` in the
   attached dataset has the **highest `<N>`** — this is *not* a fixed value like
   `ckpt_step20000.pt`; `<N>` varies with wherever the previous session stopped. Select it by
   step number, e.g. with the same `latest_checkpoint` helper used in the training notebook:
   ```python
   import glob, os, re

   def latest_checkpoint(ckpt_dir):
       paths = glob.glob(os.path.join(ckpt_dir, "ckpt_step*.pt"))
       if not paths:
           raise FileNotFoundError(f"no checkpoints in {ckpt_dir}")
       return max(paths, key=lambda p: int(re.search(r"ckpt_step(\d+)\.pt", os.path.basename(p)).group(1)))

   latest_ckpt = latest_checkpoint("/kaggle/input/<dataset-name>")
   ```
   or, from the CLI, list the dataset folder to find the highest `<N>` and copy that file:
   ```bash
   ls /kaggle/input/<dataset-name>/   # find the ckpt_step<N>.pt with the highest <N>
   cp /kaggle/input/<dataset-name>/ckpt_step<N>.pt /kaggle/working/resume_ckpt.pt
   python -m literank.cli train --scorer lite --proj-dim 768 --subset-size 100000 \
       --max-steps 40000 --checkpoint-dir /kaggle/working/ckpt_lite \
       --resume /kaggle/working/resume_ckpt.pt --device cuda
   ```
   `--resume` is passed straight to `literank.train.train`, which calls
   `literank.checkpoint.load_checkpoint` to restore model, optimizer, and AMP grad-scaler state
   and resumes counting from the saved `step`.

## Ablations

The notebook and config surface three ablations called out in the paper:

1. **LITE vs. MaxSim** — `--scorer lite` (separable row/column MLPs over the similarity
   matrix, paper Linear->activation->LayerNorm order, final learned projection to a scalar) vs.
   `--scorer maxsim` (ColBERT-style `sum_i max_j sim(q_i, d_j)` baseline, no learned
   parameters). Both share the same `DualEncoder` and training loop
   (`literank.model.build_scorer` selects the scorer from `ModelConfig.scorer`).
2. **Small-LITE projection-dimension storage** — `ModelConfig.proj_dim` controls the
   dual-encoder's output dimension `d'`. Setting `--proj-dim 128` (vs. the full `embed_dim=768`)
   shrinks the per-token embedding and therefore the serialized offline document cache written
   by `literank.encode_cache.encode_and_cache`/`save_embeddings` — directly trading retrieval
   quality for cache size, which the notebook measures by comparing the returned byte counts.
3. **Activation choice** — `ModelConfig.activation` selects the nonlinearity (`"relu"`
   (default), `"sigmoid"`, or `"gelu"`) used inside the LITE scorer's row/column MLPs
   (`literank.model._SeqMLP`). Re-run training with `ModelConfig(activation=...)` to compare.

## Known limitations

- Numbers from local or Kaggle runs are **expected to fall short of the paper's 0.393 MRR@10**
  given the reduced training subset, step budget, and single-GPU-class hardware — treat this
  repo as a correctness/qualitative reference, not a benchmark submission.
- The KL-distillation weight (`TrainConfig.kl_weight`, default `0.0`) and the activation
  default (`ModelConfig.activation`, default `"relu"`) are config-exposed best-effort choices
  inferred from the paper; both are easy to sweep via the config dataclasses if the authors'
  reference values become available.
