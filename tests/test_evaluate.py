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
