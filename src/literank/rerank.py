import torch


def rerank(scorer, query_emb, query_mask, doc_embs, doc_masks, device="cpu"):
    scorer = scorer.to(device).eval()
    n = doc_embs.shape[0]
    q = query_emb.to(device).expand(n, -1, -1)
    qm = query_mask.to(device).expand(n, -1)
    with torch.no_grad():
        scores = scorer(q, doc_embs.to(device), qm, doc_masks.to(device))
    return torch.argsort(scores, descending=True).cpu().tolist()
