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
