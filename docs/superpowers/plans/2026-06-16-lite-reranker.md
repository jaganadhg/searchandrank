# LITE Re-ranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a faithful, tested Separable-LITE document re-ranker (dual-encoder + learnable late-interaction scorer + distillation training + embedding-cache rerank + eval) that trains on a MS MARCO subset on the Kaggle free GPU and beats a MaxSim baseline.

**Architecture:** A shared HF dual-encoder produces token-level query/doc embeddings; a `LITEScorer` runs row-wise then column-wise MLPs over the sequence axes of the `Q·Dᵀ` similarity matrix and projects to a scalar. A `MaxSimScorer` baseline shares the same interface. Training distills a cross-encoder teacher with Margin-MSE(+KL) over triplets, with checkpoint/resume for Kaggle's 12 h sessions. Eval caches doc token embeddings to disk and reranks a dev subset (MRR@10, nDCG@10).

**Tech Stack:** Python 3.12, PyTorch 2.5.1+cu121, transformers, datasets, scikit-learn, numpy, pytest, ruff; uv for env management.

## Global Constraints

- Python pinned to **3.12**; `requires-python = ">=3.12,<3.13"`. (copied from spec)
- `torch==2.5.1+cu121` via the configured `pytorch-cu121` uv index/source — do **not** let resolution drift to cu130/cu128 (needs newer driver, falls back to CPU). (copied from spec)
- New package lives under `src/literank/`; old `./literank/` (Poetry) is reference only — do not modify it.
- All commands run through uv: `uv run pytest ...`, `uv run python -m literank.cli ...`.
- Paper hyperparameter defaults: `L₁(query_len)=30, L₂(doc_len)=200, d(embed_dim)=768, d'(proj_dim)=768, m₁(mlp1_hidden)=360, m₂(mlp2_hidden)=2400`.
- Tests must run **offline** (no model/dataset download). Real-model paths go behind `@pytest.mark.integration` and are exercised in the Kaggle notebook, not in CI.
- LayerNorm placement in MLP blocks follows the paper exactly: `x → Linear → activation → LayerNorm` per layer.
- TDD: write the failing test first, watch it fail, implement minimally, watch it pass, commit.

---

### Task 1: Project scaffold, build config, and run configs

**Files:**
- Modify: `pyproject.toml`
- Create: `src/literank/__init__.py`
- Create: `src/literank/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `ModelConfig`, `TrainConfig`, `DataConfig` dataclasses (see fields below).

- [ ] **Step 1: Add build + pytest config to `pyproject.toml`**

Append these sections to `pyproject.toml` (keep existing `[project]`, `[dependency-groups]`, `[tool.uv.*]`):

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/literank"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
markers = ["integration: requires network/model download (skipped by default in CI)"]

[tool.ruff]
line-length = 100
```

- [ ] **Step 2: Write the failing test**

`tests/test_config.py`:

```python
from literank.config import ModelConfig, TrainConfig, DataConfig


def test_model_config_paper_defaults():
    cfg = ModelConfig()
    assert cfg.encoder_name == "distilbert-base-uncased"
    assert (cfg.query_len, cfg.doc_len) == (30, 200)
    assert (cfg.embed_dim, cfg.proj_dim) == (768, 768)
    assert (cfg.mlp1_hidden, cfg.mlp2_hidden) == (360, 2400)
    assert cfg.activation == "relu"
    assert cfg.scorer == "lite"
    assert cfg.l2_normalize is True
    assert cfg.freeze_encoder is False


def test_train_and_data_defaults():
    t = TrainConfig()
    assert t.lr == 2.8e-5
    assert t.amp is True
    assert t.checkpoint_every == 1000
    d = DataConfig()
    assert d.teacher_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert d.subset_size == 100_000
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'literank'`.

- [ ] **Step 4: Create the package and config**

`src/literank/__init__.py`:

```python
__version__ = "0.1.0"
```

`src/literank/config.py`:

```python
from dataclasses import dataclass


@dataclass
class ModelConfig:
    encoder_name: str = "distilbert-base-uncased"
    embed_dim: int = 768        # d
    proj_dim: int = 768         # d' (< embed_dim => Small LITE)
    query_len: int = 30         # L1
    doc_len: int = 200          # L2
    mlp1_hidden: int = 360      # m1 (over query axis L1)
    mlp2_hidden: int = 2400     # m2 (over doc axis L2)
    activation: str = "relu"    # "relu" | "sigmoid" | "gelu"
    l2_normalize: bool = True
    scorer: str = "lite"        # "lite" | "maxsim"
    freeze_encoder: bool = False


@dataclass
class TrainConfig:
    lr: float = 2.8e-5
    batch_size: int = 16
    grad_accum: int = 1
    max_steps: int = 20_000
    kl_weight: float = 0.0
    amp: bool = True
    seed: int = 42
    checkpoint_every: int = 1000
    checkpoint_dir: str = "checkpoints"


@dataclass
class DataConfig:
    dataset_name: str = "microsoft/ms_marco"
    dataset_config: str = "v2.1"
    subset_size: int = 100_000
    teacher_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    teacher_max_len: int = 256
    num_dev_queries: int = 1000
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/literank/__init__.py src/literank/config.py tests/test_config.py
git commit -m "feat: package scaffold and config dataclasses"
```

---

### Task 2: Distillation losses

**Files:**
- Create: `src/literank/losses.py`
- Test: `tests/test_losses.py`

**Interfaces:**
- Produces: `margin_mse(s_pos, s_neg, t_pos, t_neg) -> Tensor`, `kl_distill(s_pos, s_neg, t_pos, t_neg) -> Tensor`, `distill_loss(s_pos, s_neg, t_pos, t_neg, kl_weight=0.0) -> Tensor`. All inputs are 1-D float tensors of equal length.

- [ ] **Step 1: Write the failing test**

`tests/test_losses.py`:

```python
import torch
from literank.losses import margin_mse, kl_distill, distill_loss


def test_margin_mse_matches_hand_value():
    s_pos = torch.tensor([3.0, 2.0])
    s_neg = torch.tensor([1.0, 1.0])
    t_pos = torch.tensor([2.0, 2.0])
    t_neg = torch.tensor([0.0, 1.0])
    # student margins [2,1], teacher margins [2,1] -> mse 0
    assert torch.allclose(margin_mse(s_pos, s_neg, t_pos, t_neg), torch.tensor(0.0))


def test_kl_is_zero_when_distributions_match():
    s = torch.tensor([2.0, -1.0])
    out = kl_distill(s, -s, s, -s)
    assert torch.allclose(out, torch.tensor(0.0), atol=1e-6)


def test_distill_loss_kl_weight_zero_equals_margin_mse():
    a = torch.randn(5); b = torch.randn(5); c = torch.randn(5); d = torch.randn(5)
    assert torch.allclose(distill_loss(a, b, c, d, kl_weight=0.0),
                          margin_mse(a, b, c, d))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_losses.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'literank.losses'`.

- [ ] **Step 3: Write minimal implementation**

`src/literank/losses.py`:

```python
import torch
import torch.nn.functional as F


def margin_mse(s_pos, s_neg, t_pos, t_neg):
    return F.mse_loss(s_pos - s_neg, t_pos - t_neg)


def kl_distill(s_pos, s_neg, t_pos, t_neg):
    student = torch.stack([s_pos, s_neg], dim=-1)
    teacher = torch.stack([t_pos, t_neg], dim=-1)
    log_p = F.log_softmax(student, dim=-1)
    q = F.softmax(teacher, dim=-1)
    return F.kl_div(log_p, q, reduction="batchmean")


def distill_loss(s_pos, s_neg, t_pos, t_neg, kl_weight=0.0):
    loss = margin_mse(s_pos, s_neg, t_pos, t_neg)
    if kl_weight > 0:
        loss = loss + kl_weight * kl_distill(s_pos, s_neg, t_pos, t_neg)
    return loss
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_losses.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/literank/losses.py tests/test_losses.py
git commit -m "feat: margin-mse and kl distillation losses"
```

---

### Task 3: LITEScorer (separable row/col MLP scorer)

**Files:**
- Create: `src/literank/model.py`
- Test: `tests/test_lite_scorer.py`

**Interfaces:**
- Produces: `LITEScorer(cfg: ModelConfig)` with `forward(q, d, q_mask=None, d_mask=None) -> Tensor[B]`. `q` is `[B, L1, d']`, `d` is `[B, L2, d']`. Masks are accepted and ignored (padded embeddings are already zeroed upstream).
- Produces: `_SeqMLP(dim, hidden, activation)` helper.

- [ ] **Step 1: Write the failing test**

`tests/test_lite_scorer.py`:

```python
import torch
from literank.config import ModelConfig
from literank.model import LITEScorer


def _cfg():
    return ModelConfig(query_len=4, doc_len=5, embed_dim=8, proj_dim=8,
                       mlp1_hidden=16, mlp2_hidden=16)


def test_output_shape_is_batch():
    cfg = _cfg()
    scorer = LITEScorer(cfg)
    q = torch.randn(3, cfg.query_len, cfg.proj_dim)
    d = torch.randn(3, cfg.doc_len, cfg.proj_dim)
    out = scorer(q, d)
    assert out.shape == (3,)


def test_gradients_reach_all_params():
    cfg = _cfg()
    scorer = LITEScorer(cfg)
    q = torch.randn(2, cfg.query_len, cfg.proj_dim)
    d = torch.randn(2, cfg.doc_len, cfg.proj_dim)
    scorer(q, d).sum().backward()
    assert all(p.grad is not None for p in scorer.parameters())


def test_accepts_and_ignores_masks():
    cfg = _cfg()
    scorer = LITEScorer(cfg)
    q = torch.randn(2, cfg.query_len, cfg.proj_dim)
    d = torch.randn(2, cfg.doc_len, cfg.proj_dim)
    qm = torch.ones(2, cfg.query_len)
    dm = torch.ones(2, cfg.doc_len)
    assert torch.allclose(scorer(q, d), scorer(q, d, qm, dm))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lite_scorer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'literank.model'`.

- [ ] **Step 3: Write minimal implementation**

`src/literank/model.py`:

```python
import torch
import torch.nn as nn

_ACT = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "gelu": nn.GELU}


class _SeqMLP(nn.Module):
    """Two-layer MLP over the last dim (size `dim`); paper order Linear->act->LN."""

    def __init__(self, dim, hidden, activation="relu"):
        super().__init__()
        act = _ACT[activation]
        self.fc1 = nn.Linear(dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.act = act()

    def forward(self, x):
        x = self.ln1(self.act(self.fc1(x)))
        x = self.ln2(self.act(self.fc2(x)))
        return x


class LITEScorer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.row_mlp = _SeqMLP(cfg.doc_len, cfg.mlp2_hidden, cfg.activation)   # over L2
        self.col_mlp = _SeqMLP(cfg.query_len, cfg.mlp1_hidden, cfg.activation)  # over L1
        self.final = nn.Linear(cfg.query_len * cfg.doc_len, 1)

    def forward(self, q, d, q_mask=None, d_mask=None):
        s = torch.einsum("bid,bjd->bij", q, d)                 # [B, L1, L2]
        s = self.row_mlp(s)                                    # MLP over L2
        s = self.col_mlp(s.transpose(1, 2)).transpose(1, 2)    # MLP over L1
        b = s.shape[0]
        return self.final(s.reshape(b, -1)).squeeze(-1)        # [B]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_lite_scorer.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/literank/model.py tests/test_lite_scorer.py
git commit -m "feat: separable LITE scorer"
```

---

### Task 4: MaxSimScorer baseline and scorer factory

**Files:**
- Modify: `src/literank/model.py`
- Test: `tests/test_maxsim_scorer.py`

**Interfaces:**
- Produces: `MaxSimScorer()` with `forward(q, d, q_mask=None, d_mask=None) -> Tensor[B]` = sum over valid query tokens of max over valid doc tokens.
- Produces: `build_scorer(cfg: ModelConfig) -> nn.Module` returning `LITEScorer` for `cfg.scorer=="lite"`, `MaxSimScorer` for `"maxsim"`.

- [ ] **Step 1: Write the failing test**

`tests/test_maxsim_scorer.py`:

```python
import torch
from literank.config import ModelConfig
from literank.model import MaxSimScorer, LITEScorer, build_scorer


def test_maxsim_matches_hand_value():
    # one query token, two doc tokens; sims [0.2, 0.9] -> max 0.9
    q = torch.tensor([[[1.0, 0.0]]])           # [1,1,2]
    d = torch.tensor([[[0.2, 0.0], [0.9, 0.0]]])  # [1,2,2]
    out = MaxSimScorer()(q, d)
    assert torch.allclose(out, torch.tensor([0.9]))


def test_maxsim_respects_doc_mask():
    q = torch.tensor([[[1.0, 0.0]]])
    d = torch.tensor([[[0.2, 0.0], [0.9, 0.0]]])
    d_mask = torch.tensor([[1.0, 0.0]])        # second doc token padded out
    out = MaxSimScorer()(q, d, d_mask=d_mask)
    assert torch.allclose(out, torch.tensor([0.2]))


def test_build_scorer_dispatch():
    assert isinstance(build_scorer(ModelConfig(scorer="maxsim")), MaxSimScorer)
    assert isinstance(build_scorer(ModelConfig(scorer="lite")), LITEScorer)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_maxsim_scorer.py -v`
Expected: FAIL with `ImportError: cannot import name 'MaxSimScorer'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/literank/model.py`:

```python
class MaxSimScorer(nn.Module):
    """ColBERT-style baseline: sum_i max_j (q_i . d_j), masked."""

    def __init__(self, cfg=None):
        super().__init__()

    def forward(self, q, d, q_mask=None, d_mask=None):
        s = torch.einsum("bid,bjd->bij", q, d)             # [B, L1, L2]
        if d_mask is not None:
            neg = torch.finfo(s.dtype).min
            s = s.masked_fill(d_mask.unsqueeze(1) == 0, neg)
        maxsim = s.max(dim=2).values                        # [B, L1]
        if q_mask is not None:
            maxsim = maxsim * q_mask
        return maxsim.sum(dim=1)                             # [B]


def build_scorer(cfg):
    if cfg.scorer == "lite":
        return LITEScorer(cfg)
    if cfg.scorer == "maxsim":
        return MaxSimScorer(cfg)
    raise ValueError(f"unknown scorer: {cfg.scorer}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_maxsim_scorer.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/literank/model.py tests/test_maxsim_scorer.py
git commit -m "feat: maxsim baseline scorer and scorer factory"
```

---

### Task 5: DualEncoder (tokenize / embed / encode)

**Files:**
- Create: `src/literank/encoder.py`
- Test: `tests/test_encoder.py`

**Interfaces:**
- Produces: `DualEncoder(cfg, encoder=None, tokenizer=None)` (nn.Module) with:
  - `tokenize(texts, max_len) -> dict` (uses HF tokenizer)
  - `embed(input_ids, attention_mask) -> (Tensor[B,L,d'], Tensor[B,L])` — applies encoder, optional projection, zeroes padded positions, optional L2-normalize.
  - `encode(texts, max_len) -> (Tensor, Tensor)` — tokenize then embed.
- Note: `embed` is unit-tested offline with a tiny injected encoder; `encode`/`tokenize` need the real tokenizer and are covered by `@pytest.mark.integration`.

- [ ] **Step 1: Write the failing test**

`tests/test_encoder.py`:

```python
import torch
import torch.nn as nn
from literank.config import ModelConfig
from literank.encoder import DualEncoder


class _TinyEncoder(nn.Module):
    """Returns object with .last_hidden_state of shape [B,L,hidden]."""

    def __init__(self, hidden):
        super().__init__()
        self.emb = nn.Embedding(100, hidden)

    def forward(self, input_ids, attention_mask=None):
        class Out:
            pass
        o = Out()
        o.last_hidden_state = self.emb(input_ids)
        return o


def _enc(proj_dim=8, normalize=False):
    cfg = ModelConfig(embed_dim=8, proj_dim=proj_dim, l2_normalize=normalize)
    return DualEncoder(cfg, encoder=_TinyEncoder(8), tokenizer=object()), cfg


def test_embed_zeros_padded_positions():
    enc, cfg = _enc()
    input_ids = torch.randint(1, 100, (2, 5))
    attn = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    emb, mask = enc.embed(input_ids, attn)
    assert emb.shape == (2, 5, cfg.proj_dim)
    assert torch.count_nonzero(emb[0, 3:]) == 0
    assert torch.count_nonzero(emb[1, 2:]) == 0
    assert torch.equal(mask, attn)


def test_projection_changes_dim():
    enc, cfg = _enc(proj_dim=4)
    emb, _ = enc.embed(torch.randint(1, 100, (1, 3)), torch.ones(1, 3, dtype=torch.long))
    assert emb.shape[-1] == 4


def test_l2_normalize_unit_rows_on_valid_tokens():
    enc, _ = _enc(normalize=True)
    emb, _ = enc.embed(torch.randint(1, 100, (1, 3)), torch.ones(1, 3, dtype=torch.long))
    norms = emb[0].norm(dim=-1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_encoder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'literank.encoder'`.

- [ ] **Step 3: Write minimal implementation**

`src/literank/encoder.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualEncoder(nn.Module):
    def __init__(self, cfg, encoder=None, tokenizer=None):
        super().__init__()
        self.cfg = cfg
        if tokenizer is None or encoder is None:
            from transformers import AutoModel, AutoTokenizer
            tokenizer = tokenizer or AutoTokenizer.from_pretrained(cfg.encoder_name)
            encoder = encoder or AutoModel.from_pretrained(cfg.encoder_name)
        self.tokenizer = tokenizer
        self.encoder = encoder
        if cfg.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.proj = nn.Linear(cfg.embed_dim, cfg.proj_dim) if cfg.proj_dim != cfg.embed_dim else None

    def tokenize(self, texts, max_len):
        return self.tokenizer(texts, padding="max_length", truncation=True,
                              max_length=max_len, return_tensors="pt")

    def embed(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        emb = out.last_hidden_state
        if self.proj is not None:
            emb = self.proj(emb)
        if self.cfg.l2_normalize:
            emb = F.normalize(emb, dim=-1)
        emb = emb * attention_mask.unsqueeze(-1).to(emb.dtype)
        return emb, attention_mask

    def encode(self, texts, max_len):
        enc = self.tokenize(texts, max_len)
        device = next(self.parameters()).device
        return self.embed(enc["input_ids"].to(device), enc["attention_mask"].to(device))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_encoder.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/literank/encoder.py tests/test_encoder.py
git commit -m "feat: dual-encoder with masking, projection, l2-normalize"
```

---

### Task 6: Ranker (encoder + scorer) and integration smoke

**Files:**
- Modify: `src/literank/model.py`
- Test: `tests/test_ranker.py`

**Interfaces:**
- Produces: `Ranker(cfg, encoder=None)` (nn.Module) with `score(queries: list[str], docs: list[str]) -> Tensor[B]`, holding a `DualEncoder` and a scorer from `build_scorer`.

- [ ] **Step 1: Write the failing test**

`tests/test_ranker.py`:

```python
import pytest
import torch
from literank.config import ModelConfig
from literank.model import Ranker


def test_score_uses_scorer_on_encoder_outputs(monkeypatch):
    cfg = ModelConfig(query_len=4, doc_len=5, embed_dim=8, proj_dim=8,
                      mlp1_hidden=16, mlp2_hidden=16)
    ranker = Ranker.__new__(Ranker)
    torch.nn.Module.__init__(ranker)

    class FakeEnc:
        cfg = cfg

        def encode(self, texts, max_len):
            b = len(texts)
            return torch.randn(b, max_len, cfg.proj_dim), torch.ones(b, max_len)

    ranker.encoder = FakeEnc()
    from literank.model import build_scorer
    ranker.scorer = build_scorer(cfg)
    out = ranker.score(["q1", "q2"], ["d1", "d2"])
    assert out.shape == (2,)


@pytest.mark.integration
def test_ranker_end_to_end_real_distilbert():
    cfg = ModelConfig()
    ranker = Ranker(cfg)
    out = ranker.score(["what is ai"], ["ai is the study of intelligent agents"])
    assert out.shape == (1,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ranker.py::test_score_uses_scorer_on_encoder_outputs -v`
Expected: FAIL with `ImportError: cannot import name 'Ranker'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/literank/model.py` (add `from literank.encoder import DualEncoder` at top of file):

```python
class Ranker(nn.Module):
    def __init__(self, cfg, encoder=None):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder or DualEncoder(cfg)
        self.scorer = build_scorer(cfg)

    def score(self, queries, docs):
        q, qm = self.encoder.encode(queries, self.cfg.query_len)
        d, dm = self.encoder.encode(docs, self.cfg.doc_len)
        return self.scorer(q, d, qm, dm)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ranker.py::test_score_uses_scorer_on_encoder_outputs -v`
Expected: PASS (1 passed). The integration test is skipped by default (`-m "not integration"`).

- [ ] **Step 5: Commit**

```bash
git add src/literank/model.py tests/test_ranker.py
git commit -m "feat: ranker combining encoder and scorer"
```

---

### Task 7: Triplet dataset, collation, and MS MARCO builder

**Files:**
- Create: `src/literank/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces: `TripletDataset(triplets: list[dict])` (each dict: `query, pos, neg, t_pos, t_neg`).
- Produces: `collate_triplets(batch) -> dict` with keys `query, pos, neg` (lists) and `t_pos, t_neg` (float tensors).
- Produces: `build_msmarco_triplets(data_cfg) -> list[dict]` (no teacher scores yet: `t_pos=t_neg=0.0`), `@pytest.mark.integration` only.

- [ ] **Step 1: Write the failing test**

`tests/test_data.py`:

```python
import torch
from literank.data import TripletDataset, collate_triplets


def _triplets():
    return [
        {"query": "q1", "pos": "p1", "neg": "n1", "t_pos": 2.0, "t_neg": 0.0},
        {"query": "q2", "pos": "p2", "neg": "n2", "t_pos": 1.5, "t_neg": -1.0},
    ]


def test_dataset_len_and_getitem():
    ds = TripletDataset(_triplets())
    assert len(ds) == 2
    assert ds[0]["query"] == "q1"


def test_collate_shapes_and_types():
    batch = collate_triplets(_triplets())
    assert batch["query"] == ["q1", "q2"]
    assert batch["neg"] == ["n1", "n2"]
    assert torch.allclose(batch["t_pos"], torch.tensor([2.0, 1.5]))
    assert batch["t_neg"].dtype == torch.float32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'literank.data'`.

- [ ] **Step 3: Write minimal implementation**

`src/literank/data.py`:

```python
import random
import torch
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]


def collate_triplets(batch):
    return {
        "query": [b["query"] for b in batch],
        "pos": [b["pos"] for b in batch],
        "neg": [b["neg"] for b in batch],
        "t_pos": torch.tensor([b["t_pos"] for b in batch], dtype=torch.float32),
        "t_neg": torch.tensor([b["t_neg"] for b in batch], dtype=torch.float32),
    }


def build_msmarco_triplets(data_cfg, seed=42):
    """Build (query, pos, neg) triplets from MS MARCO; teacher scores set later."""
    from datasets import load_dataset
    rng = random.Random(seed)
    ds = load_dataset(data_cfg.dataset_name, data_cfg.dataset_config, split="train")
    ds = ds.select(range(min(data_cfg.subset_size, len(ds))))
    triplets = []
    for rec in ds:
        passages = rec["passages"]["passage_text"]
        selected = rec["passages"]["is_selected"]
        pos_idx = [i for i, s in enumerate(selected) if s == 1]
        neg_idx = [i for i, s in enumerate(selected) if s == 0]
        if not pos_idx or not neg_idx:
            continue
        p = passages[rng.choice(pos_idx)]
        n = passages[rng.choice(neg_idx)]
        triplets.append({"query": rec["query"], "pos": p, "neg": n,
                         "t_pos": 0.0, "t_neg": 0.0})
    return triplets
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_data.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/literank/data.py tests/test_data.py
git commit -m "feat: triplet dataset, collation, ms marco builder"
```

---

### Task 8: Cross-encoder teacher and score caching

**Files:**
- Create: `src/literank/teacher.py`
- Test: `tests/test_teacher.py`

**Interfaces:**
- Produces: `CrossEncoderTeacher(name, device="cpu", max_len=256)` with `score(queries, passages, batch_size=32) -> list[float]` (`@pytest.mark.integration`).
- Produces: `add_teacher_scores(triplets, teacher, batch_size=32) -> list[dict]` — fills `t_pos`/`t_neg` in place using `teacher.score`.
- Produces: `cache_teacher_scores(triplets, teacher, path) -> list[dict]` — returns cached scores if `path` exists (JSON), else computes, writes, returns.

- [ ] **Step 1: Write the failing test**

`tests/test_teacher.py`:

```python
import json
from literank.teacher import add_teacher_scores, cache_teacher_scores


class StubTeacher:
    def __init__(self):
        self.calls = 0

    def score(self, queries, passages, batch_size=32):
        self.calls += 1
        return [float(len(p)) for p in passages]


def _triplets():
    return [{"query": "q", "pos": "abcd", "neg": "ab", "t_pos": 0.0, "t_neg": 0.0}]


def test_add_teacher_scores_fills_margins():
    out = add_teacher_scores(_triplets(), StubTeacher())
    assert out[0]["t_pos"] == 4.0 and out[0]["t_neg"] == 2.0


def test_cache_round_trips_and_skips_recompute(tmp_path):
    teacher = StubTeacher()
    path = tmp_path / "scores.json"
    cache_teacher_scores(_triplets(), teacher, str(path))
    assert path.exists()
    first_calls = teacher.calls
    cache_teacher_scores(_triplets(), teacher, str(path))   # should reuse file
    assert teacher.calls == first_calls
    data = json.loads(path.read_text())
    assert data[0]["t_pos"] == 4.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_teacher.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'literank.teacher'`.

- [ ] **Step 3: Write minimal implementation**

`src/literank/teacher.py`:

```python
import json
import os


class CrossEncoderTeacher:
    def __init__(self, name, device="cpu", max_len=256):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        self.device = device
        self.max_len = max_len

    def score(self, queries, passages, batch_size=32):
        out = []
        for i in range(0, len(queries), batch_size):
            q = queries[i:i + batch_size]
            p = passages[i:i + batch_size]
            enc = self.tokenizer(q, p, padding=True, truncation=True,
                                 max_length=self.max_len, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                logits = self.model(**enc).logits.squeeze(-1)
            out.extend(logits.detach().cpu().tolist())
        return out


def add_teacher_scores(triplets, teacher, batch_size=32):
    queries = [t["query"] for t in triplets]
    pos = [t["pos"] for t in triplets]
    neg = [t["neg"] for t in triplets]
    t_pos = teacher.score(queries, pos, batch_size=batch_size)
    t_neg = teacher.score(queries, neg, batch_size=batch_size)
    for t, sp, sn in zip(triplets, t_pos, t_neg):
        t["t_pos"], t["t_neg"] = float(sp), float(sn)
    return triplets


def cache_teacher_scores(triplets, teacher, path, batch_size=32):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    out = add_teacher_scores(triplets, teacher, batch_size=batch_size)
    with open(path, "w") as f:
        json.dump(out, f)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_teacher.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/literank/teacher.py tests/test_teacher.py
git commit -m "feat: cross-encoder teacher and score caching"
```

---

### Task 9: Checkpoint save/load

**Files:**
- Create: `src/literank/checkpoint.py`
- Test: `tests/test_checkpoint.py`

**Interfaces:**
- Produces: `save_checkpoint(path, model, optimizer, scaler, step, config)`.
- Produces: `load_checkpoint(path, model, optimizer=None, scaler=None, map_location="cpu") -> int` (returns saved step).

- [ ] **Step 1: Write the failing test**

`tests/test_checkpoint.py`:

```python
import torch
import torch.nn as nn
from literank.checkpoint import save_checkpoint, load_checkpoint


def test_save_then_load_restores_state_and_step(tmp_path):
    model = nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(3, 4)).sum().backward()
    opt.step()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(str(path), model, opt, None, step=123, config={"a": 1})

    model2 = nn.Linear(4, 2)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    step = load_checkpoint(str(path), model2, opt2)
    assert step == 123
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_checkpoint.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'literank.checkpoint'`.

- [ ] **Step 3: Write minimal implementation**

`src/literank/checkpoint.py`:

```python
import torch


def save_checkpoint(path, model, optimizer, scaler, step, config):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "config": config,
    }, path)


def load_checkpoint(path, model, optimizer=None, scaler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt["step"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_checkpoint.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/literank/checkpoint.py tests/test_checkpoint.py
git commit -m "feat: checkpoint save/load"
```

---

### Task 10: Training loop (train_step, train, resume)

**Files:**
- Create: `src/literank/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `distill_loss` (Task 2), `TripletDataset`/`collate_triplets` (Task 7), `save_checkpoint`/`load_checkpoint` (Task 9).
- Produces: `train_step(model, batch, optimizer, scaler, kl_weight, grad_accum, do_step, device) -> float` (returns the unscaled loss value).
- Produces: `train(model_cfg, train_cfg, triplets, model=None, resume=None, device="cpu") -> nn.Module`. `model` must expose `.score(queries, docs)`; defaults to `Ranker(model_cfg)`.

- [ ] **Step 1: Write the failing test**

`tests/test_train.py`:

```python
import torch
import torch.nn as nn
from literank.config import ModelConfig, TrainConfig
from literank.train import train


class ToyRanker(nn.Module):
    """Text-driven, learnable, no tokenizer — for loop/checkpoint tests."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))

    def score(self, queries, docs):
        base = torch.tensor([float(len(d)) for d in docs])
        return self.w * base


def _triplets():
    # pos text longer than neg; teacher prefers pos
    return [{"query": "q", "pos": "aaaaaa", "neg": "aa", "t_pos": 5.0, "t_neg": 0.0}] * 8


def test_training_decreases_loss(tmp_path):
    tcfg = TrainConfig(batch_size=4, max_steps=30, amp=False, seed=0,
                       checkpoint_every=1000, checkpoint_dir=str(tmp_path))
    model = ToyRanker()
    # capture initial loss
    train(ModelConfig(), tcfg, _triplets(), model=model, device="cpu")
    # after training, weight should have moved off zero toward positive margin
    assert model.w.item() != 0.0


def test_checkpoint_and_resume(tmp_path):
    tcfg = TrainConfig(batch_size=4, max_steps=5, amp=False, seed=0,
                       checkpoint_every=5, checkpoint_dir=str(tmp_path))
    train(ModelConfig(), tcfg, _triplets(), model=ToyRanker(), device="cpu")
    ckpt = tmp_path / "ckpt_step5.pt"
    assert ckpt.exists()
    # resume continues without error and runs more steps
    tcfg2 = TrainConfig(batch_size=4, max_steps=10, amp=False, seed=0,
                        checkpoint_every=5, checkpoint_dir=str(tmp_path))
    train(ModelConfig(), tcfg2, _triplets(), model=ToyRanker(),
          resume=str(ckpt), device="cpu")
    assert (tmp_path / "ckpt_step10.pt").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'literank.train'`.

- [ ] **Step 3: Write minimal implementation**

`src/literank/train.py`:

```python
import os
import torch
from torch.utils.data import DataLoader
from literank.data import TripletDataset, collate_triplets
from literank.losses import distill_loss
from literank.checkpoint import save_checkpoint, load_checkpoint


def train_step(model, batch, optimizer, scaler, kl_weight, grad_accum, do_step, device):
    use_amp = scaler is not None and scaler.is_enabled()
    with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=use_amp):
        s_pos = model.score(batch["query"], batch["pos"])
        s_neg = model.score(batch["query"], batch["neg"])
        loss = distill_loss(s_pos, s_neg,
                            batch["t_pos"].to(s_pos.device),
                            batch["t_neg"].to(s_neg.device),
                            kl_weight=kl_weight)
    scaled = loss / grad_accum
    if use_amp:
        scaler.scale(scaled).backward()
    else:
        scaled.backward()
    if do_step:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    return float(loss.detach())


def train(model_cfg, train_cfg, triplets, model=None, resume=None, device="cpu"):
    torch.manual_seed(train_cfg.seed)
    if model is None:
        from literank.model import Ranker
        model = Ranker(model_cfg)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=train_cfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg.amp and device == "cuda")

    start_step = 0
    if resume:
        start_step = load_checkpoint(resume, model, optimizer, scaler, map_location=device)

    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    loader = DataLoader(TripletDataset(triplets), batch_size=train_cfg.batch_size,
                        shuffle=True, collate_fn=collate_triplets)
    model.train()
    optimizer.zero_grad()
    step = start_step
    while step < train_cfg.max_steps:
        for batch in loader:
            step += 1
            do_step = (step % train_cfg.grad_accum == 0)
            train_step(model, batch, optimizer, scaler, train_cfg.kl_weight,
                       train_cfg.grad_accum, do_step, device)
            if step % train_cfg.checkpoint_every == 0 or step >= train_cfg.max_steps:
                save_checkpoint(os.path.join(train_cfg.checkpoint_dir, f"ckpt_step{step}.pt"),
                                model, optimizer, scaler, step, vars(model_cfg))
            if step >= train_cfg.max_steps:
                break
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_train.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/literank/train.py tests/test_train.py
git commit -m "feat: distillation training loop with checkpoint/resume"
```

---

### Task 11: Offline embedding cache (storage lever)

**Files:**
- Create: `src/literank/encode_cache.py`
- Test: `tests/test_encode_cache.py`

**Interfaces:**
- Produces: `save_embeddings(embs, masks, path) -> int` (bytes on disk).
- Produces: `load_embeddings(path) -> (Tensor, Tensor)`.
- Produces: `encode_and_cache(encoder, docs, max_len, path, batch_size=32) -> int` (`@pytest.mark.integration`).

- [ ] **Step 1: Write the failing test**

`tests/test_encode_cache.py`:

```python
import torch
from literank.encode_cache import save_embeddings, load_embeddings


def test_round_trip(tmp_path):
    embs = torch.randn(4, 5, 8)
    masks = torch.ones(4, 5)
    path = tmp_path / "cache.pt"
    save_embeddings(embs, masks, str(path))
    e, m = load_embeddings(str(path))
    assert torch.allclose(e, embs) and torch.allclose(m, masks)


def test_smaller_projection_dim_uses_less_storage(tmp_path):
    big = save_embeddings(torch.randn(10, 20, 768), torch.ones(10, 20),
                          str(tmp_path / "big.pt"))
    small = save_embeddings(torch.randn(10, 20, 128), torch.ones(10, 20),
                            str(tmp_path / "small.pt"))
    assert small < big   # demonstrates the Small-LITE storage lever
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_encode_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'literank.encode_cache'`.

- [ ] **Step 3: Write minimal implementation**

`src/literank/encode_cache.py`:

```python
import os
import torch


def save_embeddings(embs, masks, path):
    torch.save({"emb": embs.cpu().contiguous(), "mask": masks.cpu().contiguous()}, path)
    return os.path.getsize(path)


def load_embeddings(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["emb"], data["mask"]


def encode_and_cache(encoder, docs, max_len, path, batch_size=32):
    encoder.eval()
    embs, masks = [], []
    with torch.no_grad():
        for i in range(0, len(docs), batch_size):
            e, m = encoder.encode(docs[i:i + batch_size], max_len)
            embs.append(e.cpu())
            masks.append(m.cpu())
    return save_embeddings(torch.cat(embs), torch.cat(masks), path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_encode_cache.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/literank/encode_cache.py tests/test_encode_cache.py
git commit -m "feat: offline embedding cache with storage lever"
```

---

### Task 12: Reranking with cached embeddings

**Files:**
- Create: `src/literank/rerank.py`
- Test: `tests/test_rerank.py`

**Interfaces:**
- Consumes: a scorer with `forward(q, d, q_mask, d_mask)` (Tasks 3/4).
- Produces: `rerank(scorer, query_emb, query_mask, doc_embs, doc_masks, device="cpu") -> list[int]` — descending-score order of candidate indices. `query_emb` is `[1,L1,d']`, `doc_embs` is `[N,L2,d']`.

- [ ] **Step 1: Write the failing test**

`tests/test_rerank.py`:

```python
import torch
from literank.model import MaxSimScorer
from literank.rerank import rerank


def test_rerank_orders_by_score():
    # query token aligned with doc 1's token -> doc 1 ranks first
    query_emb = torch.tensor([[[1.0, 0.0]]])            # [1,1,2]
    query_mask = torch.ones(1, 1)
    doc_embs = torch.tensor([
        [[0.1, 0.0]],   # doc 0 weak
        [[0.9, 0.0]],   # doc 1 strong
        [[0.5, 0.0]],   # doc 2 medium
    ])                                                   # [3,1,2]
    doc_masks = torch.ones(3, 1)
    order = rerank(MaxSimScorer(), query_emb, query_mask, doc_embs, doc_masks)
    assert order == [1, 2, 0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rerank.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'literank.rerank'`.

- [ ] **Step 3: Write minimal implementation**

`src/literank/rerank.py`:

```python
import torch


def rerank(scorer, query_emb, query_mask, doc_embs, doc_masks, device="cpu"):
    scorer = scorer.to(device).eval()
    n = doc_embs.shape[0]
    q = query_emb.to(device).expand(n, -1, -1)
    qm = query_mask.to(device).expand(n, -1)
    with torch.no_grad():
        scores = scorer(q, doc_embs.to(device), qm, doc_masks.to(device))
    return torch.argsort(scores, descending=True).cpu().tolist()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rerank.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/literank/rerank.py tests/test_rerank.py
git commit -m "feat: rerank candidates with cached embeddings"
```

---

### Task 13: Evaluation metrics (MRR@10, nDCG@10)

**Files:**
- Create: `src/literank/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Produces: `mrr_at_k(ranked_relevances, k=10) -> float`, `ndcg_at_k(ranked_relevances, k=10) -> float`. `ranked_relevances` is a list (per query) of relevance grades (0/1) in ranked order.

- [ ] **Step 1: Write the failing test**

`tests/test_evaluate.py`:

```python
import math
from literank.evaluate import mrr_at_k, ndcg_at_k


def test_mrr_first_relevant_at_rank_two():
    assert mrr_at_k([[0, 1, 0, 0]], k=10) == 0.5


def test_mrr_no_relevant_in_topk_is_zero():
    assert mrr_at_k([[0, 0, 0]], k=2) == 0.0


def test_mrr_averages_across_queries():
    assert mrr_at_k([[1, 0], [0, 0, 1]], k=10) == (1.0 + 1 / 3) / 2


def test_ndcg_perfect_ranking_is_one():
    assert abs(ndcg_at_k([[1, 0, 0]], k=10) - 1.0) < 1e-9


def test_ndcg_relevant_at_rank_two():
    # DCG = 1/log2(3); IDCG = 1/log2(2)=1 -> nDCG = 1/log2(3)
    assert abs(ndcg_at_k([[0, 1]], k=10) - 1 / math.log2(3)) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'literank.evaluate'`.

- [ ] **Step 3: Write minimal implementation**

`src/literank/evaluate.py`:

```python
import math


def mrr_at_k(ranked_relevances, k=10):
    if not ranked_relevances:
        return 0.0
    total = 0.0
    for rels in ranked_relevances:
        rr = 0.0
        for i, rel in enumerate(rels[:k]):
            if rel > 0:
                rr = 1.0 / (i + 1)
                break
        total += rr
    return total / len(ranked_relevances)


def _dcg(rels, k):
    return sum((rel / math.log2(i + 2)) for i, rel in enumerate(rels[:k]))


def ndcg_at_k(ranked_relevances, k=10):
    if not ranked_relevances:
        return 0.0
    total = 0.0
    for rels in ranked_relevances:
        ideal = sorted(rels, reverse=True)
        idcg = _dcg(ideal, k)
        total += (_dcg(rels, k) / idcg) if idcg > 0 else 0.0
    return total / len(ranked_relevances)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/literank/evaluate.py tests/test_evaluate.py
git commit -m "feat: MRR@10 and nDCG@10 metrics"
```

---

### Task 14: CLI (train / encode / rerank / eval)

**Files:**
- Create: `src/literank/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: all prior modules.
- Produces: `build_parser() -> argparse.ArgumentParser` with subcommands `train` (flags `--scorer`, `--resume`, `--proj-dim`, `--max-steps`, `--subset-size`, `--checkpoint-dir`), `encode`, `rerank`, `eval`. `main(argv=None)` dispatches.

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:

```python
from literank.cli import build_parser


def test_train_subcommand_parses_flags():
    p = build_parser()
    args = p.parse_args(["train", "--scorer", "maxsim", "--max-steps", "50",
                         "--proj-dim", "128", "--resume", "ck.pt"])
    assert args.command == "train"
    assert args.scorer == "maxsim"
    assert args.max_steps == 50
    assert args.proj_dim == 128
    assert args.resume == "ck.pt"


def test_subcommands_exist():
    p = build_parser()
    for cmd in ["train", "encode", "rerank", "eval"]:
        assert p.parse_args([cmd]).command == cmd
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'literank.cli'`.

- [ ] **Step 3: Write minimal implementation**

`src/literank/cli.py`:

```python
import argparse
from literank.config import ModelConfig, TrainConfig, DataConfig


def build_parser():
    p = argparse.ArgumentParser(prog="literank")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train a ranker via distillation")
    t.add_argument("--scorer", choices=["lite", "maxsim"], default="lite")
    t.add_argument("--proj-dim", type=int, default=ModelConfig().proj_dim)
    t.add_argument("--max-steps", type=int, default=TrainConfig().max_steps)
    t.add_argument("--subset-size", type=int, default=DataConfig().subset_size)
    t.add_argument("--checkpoint-dir", default=TrainConfig().checkpoint_dir)
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
    # Dispatch is wired to module entrypoints; the Kaggle notebook drives the
    # full train->encode->rerank->eval flow. Heavy paths require a GPU + data.
    if args.command == "train":
        from literank.config import ModelConfig, TrainConfig, DataConfig
        from literank.data import build_msmarco_triplets
        from literank.teacher import CrossEncoderTeacher, cache_teacher_scores
        from literank.train import train
        mcfg = ModelConfig(scorer=args.scorer, proj_dim=args.proj_dim)
        tcfg = TrainConfig(max_steps=args.max_steps, checkpoint_dir=args.checkpoint_dir)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run full suite + ruff**

Run: `uv run pytest -m "not integration" -q && uv run ruff check src tests`
Expected: all tests pass; ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/literank/cli.py tests/test_cli.py
git commit -m "feat: CLI entrypoints for train/encode/rerank/eval"
```

---

### Task 15: Kaggle run notebook + README

**Files:**
- Create: `notebooks/literank_kaggle.ipynb`
- Create: `README_LITERANK.md`

**Interfaces:** none (run artifact). This task is verified by running it on Kaggle, not by unit tests.

- [ ] **Step 1: Create the notebook**

Create `notebooks/literank_kaggle.ipynb` with these cells (as a JSON notebook; each bullet is one code cell):

1. Markdown: title + instructions ("Enable GPU T4×2 and Internet in the Kaggle sidebar; expect multi-session training via checkpoint/resume").
2. `!pip install -q -e .` (after `%cd /kaggle/working/<repo>`), or `!pip install torch transformers datasets scikit-learn` if not installing the package.
3. Smoke test: `!uv run pytest -m "not integration" -q` (or `!pytest -m "not integration" -q`).
4. Train (subset, LITE):
   ```python
   !python -m literank.cli train --scorer lite --proj-dim 768 \
       --subset-size 100000 --max-steps 20000 \
       --checkpoint-dir /kaggle/working/ckpt_lite --device cuda
   ```
5. Train baseline (MaxSim) for the ablation:
   ```python
   !python -m literank.cli train --scorer maxsim --subset-size 100000 \
       --max-steps 20000 --checkpoint-dir /kaggle/working/ckpt_maxsim --device cuda
   ```
6. Small-LITE storage ablation: rerun encode with `--proj-dim 128` and compare cache bytes (`encode_and_cache` return value) against `--proj-dim 768`.
7. Encode dev docs → cache; rerank dev subset; compute MRR@10/nDCG@10 with `literank.evaluate`.
8. Markdown: how to save `/kaggle/working/ckpt_*` as a Kaggle Dataset and pass it to `--resume` in the next session.

- [ ] **Step 2: Create the README**

`README_LITERANK.md` documenting: project goal (faithful LITE reference, Kaggle target), the uv setup (`uv sync`), how to run tests (`uv run pytest -m "not integration"`), the CLI commands, the Kaggle checkpoint→Dataset→resume loop, and the three ablations (LITE vs MaxSim; Small-LITE proj-dim storage; activation). State plainly that local numbers will be below the paper's 0.393 and that the deliverable is qualitative reproduction.

- [ ] **Step 3: Commit**

```bash
git add notebooks/literank_kaggle.ipynb README_LITERANK.md
git commit -m "docs: kaggle run notebook and project readme"
```

---

## Self-Review

**Spec coverage:**
- DualEncoder (configurable, default DistilBERT, mask, projection, l2-norm) → Task 5. ✓
- LITEScorer (separable row/col, paper LN order, final projection) → Task 3. ✓
- MaxSim baseline + factory → Task 4. ✓
- Ranker (encoder+scorer, fine-tuned default) → Task 6. ✓
- Teacher + score caching → Task 8. ✓
- Triplet data + MS MARCO builder → Task 7. ✓
- Losses (Margin-MSE + KL, λ) → Task 2. ✓
- Training + AMP + grad-accum + checkpoint/resume → Tasks 9, 10. ✓
- Offline embedding cache + storage lever → Task 11. ✓
- Rerank → Task 12. ✓
- Eval MRR@10/nDCG@10 → Task 13. ✓
- CLI → Task 14. ✓
- Kaggle notebook + README → Task 15. ✓
- Config with paper hyperparameters → Task 1. ✓
- Ablations (LITE vs MaxSim; Small-LITE proj; activation) → Tasks 4, 11, 15 + config. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. `cli.py` non-train commands intentionally defer to the notebook (documented), which is a real decision, not a placeholder — the train path is fully wired.

**Type consistency:** Scorer signature `forward(q, d, q_mask=None, d_mask=None)` is identical across `LITEScorer`, `MaxSimScorer`, and the `rerank` consumer. `Ranker.score(queries, docs)` matches the `train_step` usage and the `ToyRanker` test stub. `save_checkpoint`/`load_checkpoint` signatures match `train` usage. Config field names (`proj_dim`, `query_len`, `doc_len`, `mlp1_hidden`, `mlp2_hidden`, `scorer`) are used consistently across tasks.

**Known inferred details (from spec):** activation σ (default ReLU) and KL weighting (default λ=0) remain marked for verification against authors' reference code — both are config-exposed.
