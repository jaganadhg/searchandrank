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
        def __init__(self, cfg):
            self.cfg = cfg

        def encode(self, texts, max_len):
            b = len(texts)
            return torch.randn(b, max_len, self.cfg.proj_dim), torch.ones(b, max_len)

    ranker.cfg = cfg
    ranker.encoder = FakeEnc(cfg)
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
