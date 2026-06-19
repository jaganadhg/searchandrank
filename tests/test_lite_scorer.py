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
