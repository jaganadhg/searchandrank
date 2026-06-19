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
