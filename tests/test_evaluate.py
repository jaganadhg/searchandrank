import math
import torch
from literank.evaluate import mrr_at_k, ndcg_at_k, evaluate_ranker_mrr


class _FakeModel:
    training = False

    def eval(self):
        ...

    def train(self):
        ...

    def score(self, queries, docs):
        return torch.tensor([float(len(d)) for d in docs])


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


def test_evaluate_ranker_mrr_relevant_doc_ranked_first():
    eval_set = [{"query": "q", "docs": ["x", "longer"], "labels": [0, 1]}]
    assert evaluate_ranker_mrr(_FakeModel(), eval_set, k=10) == 1.0


def test_evaluate_ranker_mrr_relevant_doc_ranked_second():
    # relevant doc is "x" (shorter) -> scores lower than "longer" -> ranked 2nd -> MRR 0.5
    eval_set = [{"query": "q", "docs": ["longer", "x"], "labels": [0, 1]}]
    assert evaluate_ranker_mrr(_FakeModel(), eval_set, k=10) == 0.5


def test_evaluate_ranker_mrr_averages_across_examples():
    eval_set = [
        {"query": "q", "docs": ["x", "longer"], "labels": [0, 1]},
        {"query": "q", "docs": ["longer", "x"], "labels": [0, 1]},
    ]
    assert evaluate_ranker_mrr(_FakeModel(), eval_set, k=10) == 0.75


def test_ndcg_multi_query_average_and_k_truncation():
    # query 1: relevant at rank 2, no truncation (k=2 covers it)
    #   DCG = 1/log2(3); IDCG = 1/log2(2) = 1 -> nDCG = 1/log2(3)
    # query 2: relevant docs at rank 1 and rank 3; k=2 truncates the rank-3 hit
    #   DCG@2 = 1/log2(2) = 1; IDCG@2 = 1/log2(2) + 1/log2(3) = 1 + 1/log2(3)
    rels = [[0, 1, 0, 0], [1, 0, 1, 0]]
    k = 2

    ndcg1 = 1 / math.log2(3)
    ndcg2 = 1.0 / (1.0 + 1 / math.log2(3))
    expected_avg = (ndcg1 + ndcg2) / 2

    assert abs(ndcg_at_k(rels, k=k) - expected_avg) < 1e-9

    # the rank-3 relevant doc in query 2 is beyond k and contributes 0:
    # truncating to k=2 must differ from scoring the full list (no truncation).
    untruncated_ndcg2 = ndcg_at_k([rels[1]], k=10)
    truncated_ndcg2 = ndcg_at_k([rels[1]], k=k)
    assert truncated_ndcg2 < untruncated_ndcg2
