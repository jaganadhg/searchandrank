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
