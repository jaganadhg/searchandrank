import math
import torch


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


def evaluate_ranker_mrr(model, eval_set, k=10, device="cpu"):
    """Dev MRR@k using model.score(queries, docs): score each example's docs, sort
    descending, build the ranked relevance list, return mean MRR@k."""
    was_training = model.training
    model.eval()
    ranked = []
    with torch.no_grad():
        for ex in eval_set:
            docs = ex["docs"]
            if not docs:
                continue
            scores = model.score([ex["query"]] * len(docs), docs)
            order = torch.argsort(scores, descending=True).tolist()
            ranked.append([ex["labels"][i] for i in order])
    if was_training:
        model.train()
    return mrr_at_k(ranked, k)


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
