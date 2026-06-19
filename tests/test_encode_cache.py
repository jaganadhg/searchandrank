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
