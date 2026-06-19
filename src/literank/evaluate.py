import math


def mrr_at_k(ranked_relevances, k=10):
    if not ranked_relevances:
        return 0.0
    total = 0.0
    for rels in ranked_relevances:
        rr = 0.0
        for i, rel in enumerate(rels[:k]):
            if rel > 0:
                rr = 1.0 / (i + 1)
                break
        total += rr
    return total / len(ranked_relevances)


def _dcg(rels, k):
    return sum((rel / math.log2(i + 2)) for i, rel in enumerate(rels[:k]))


def ndcg_at_k(ranked_relevances, k=10):
    if not ranked_relevances:
        return 0.0
    total = 0.0
    for rels in ranked_relevances:
        ideal = sorted(rels, reverse=True)
        idcg = _dcg(ideal, k)
        total += (_dcg(rels, k) / idcg) if idcg > 0 else 0.0
    return total / len(ranked_relevances)
