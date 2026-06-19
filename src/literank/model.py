import torch
import torch.nn as nn
from literank.encoder import DualEncoder

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
