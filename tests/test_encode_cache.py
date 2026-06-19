import torch
from literank.encode_cache import save_embeddings, load_embeddings


class _FakeEncoder:
    def __init__(self, dim):
        self.dim = dim

    def eval(self):
        return self

    def encode(self, texts, max_len):
        n = len(texts)
        # deterministic per-call embeddings so concatenation order is checkable
        emb = torch.ones(n, max_len, self.dim)
        mask = torch.ones(n, max_len)
        return emb, mask


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


def test_encode_and_cache_batches_and_concatenates(tmp_path):
    from literank.encode_cache import encode_and_cache, load_embeddings
    docs = [f"doc{i}" for i in range(5)]   # 5 docs
    max_len, dim = 7, 4
    path = tmp_path / "docs.pt"
    size = encode_and_cache(_FakeEncoder(dim), docs, max_len, str(path), batch_size=2)
    assert size > 0
    emb, mask = load_embeddings(str(path))
    # 5 docs across batches of 2 (2+2+1) must concatenate to N=5 on dim 0
    assert emb.shape == (5, max_len, dim)
    assert mask.shape == (5, max_len)
