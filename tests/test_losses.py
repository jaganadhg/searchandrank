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
