"""Paper-protocol eval: rerank BM25 top-1000 on MS MARCO passage dev (small).

For each judged dev query, BM25 retrieves the top-`k_bm25` passages (via a Pyserini
prebuilt Lucene index); the model reranks them; we report MRR@10 / nDCG@10 against the
official qrels. This is the harder, paper-comparable setup (unlike the ~10-candidate
dev eval in `eval_local.py`, which saturates).

Requirements: `pyserini` (BM25 + prebuilt index + topics/qrels) and a JDK (Java 11/21).
GPU strongly recommended — this encodes up to `num_queries * k_bm25` passages.

Examples:
    # cheap subset first (a few hundred queries)
    uv run python scripts/eval_bm25_top1000.py --ckpt kaggle_res_v4/ckpt_lite_big/best.pt --num-queries 500
    # full dev-small (6980 queries) — multi-hour GPU job
    uv run python scripts/eval_bm25_top1000.py --ckpt kaggle_res_v4/ckpt_lite_big/best.pt --num-queries -1

Install Pyserini:  uv sync --extra bm25   (or, on Kaggle:  pip install pyserini)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from literank.config import ModelConfig
from literank.model import Ranker
from literank.checkpoint import load_checkpoint
from literank.rerank import rerank
from literank.evaluate import mrr_at_k, ndcg_at_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to a trained checkpoint (.pt)")
    ap.add_argument("--num-queries", type=int, default=500,
                    help="number of judged dev queries to eval; -1 = all (~6980)")
    ap.add_argument("--k-bm25", type=int, default=1000, help="BM25 candidates per query")
    ap.add_argument("--index", default="msmarco-v1-passage", help="Pyserini prebuilt index")
    ap.add_argument("--topics", default="msmarco-passage-dev-subset")
    ap.add_argument("--qrels", default="msmarco-passage-dev-subset")
    ap.add_argument("--batch-size", type=int, default=64, help="passage encoding batch size")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from pyserini.search.lucene import LuceneSearcher
    from pyserini.search import get_topics, get_qrels

    searcher = LuceneSearcher.from_prebuilt_index(args.index)
    topics = get_topics(args.topics)
    qrels = get_qrels(args.qrels)

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**blob["config"])
    ranker = Ranker(cfg).to(args.device)
    load_checkpoint(args.ckpt, ranker, map_location=args.device)
    ranker.eval()

    qids = [q for q in topics if q in qrels]          # only judged queries
    if args.num_queries > 0:
        qids = qids[: args.num_queries]
    print(f"scorer={cfg.scorer} | {len(qids)} dev queries | BM25 top-{args.k_bm25} | device={args.device}")

    def passage_text(docid):
        doc = searcher.doc(docid)
        if doc is None:
            return ""
        raw = doc.raw()
        try:
            return json.loads(raw)["contents"]
        except Exception:
            return raw

    @torch.no_grad()
    def encode_docs(passages):
        embs, masks = [], []
        for i in range(0, len(passages), args.batch_size):
            e, m = ranker.encoder.encode(passages[i : i + args.batch_size], cfg.doc_len)
            embs.append(e)
            masks.append(m)
        return torch.cat(embs), torch.cat(masks)

    ranked = []
    for n, qid in enumerate(qids, 1):
        query = topics[qid]["title"]
        hits = searcher.search(query, k=args.k_bm25)
        docids = [h.docid for h in hits]
        if not docids:
            continue
        passages = [passage_text(d) for d in docids]
        with torch.no_grad():
            doc_embs, doc_masks = encode_docs(passages)
            q_emb, q_mask = ranker.encoder.encode([query], cfg.query_len)
        order = rerank(ranker.scorer, q_emb, q_mask, doc_embs, doc_masks, device=args.device)
        rel = qrels[qid]
        ranked.append([1 if rel.get(docids[i], 0) > 0 else 0 for i in order])
        if n % 50 == 0:
            print(f"  {n}/{len(qids)} | running MRR@10 {mrr_at_k(ranked, 10):.4f}")

    print(f"\nBM25 top-{args.k_bm25} rerank over {len(ranked)} queries:")
    print(f"  MRR@10:  {mrr_at_k(ranked, 10):.4f}")
    print(f"  nDCG@10: {ndcg_at_k(ranked, 10):.4f}")


if __name__ == "__main__":
    main()
