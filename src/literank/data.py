import random
import torch
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]


def collate_triplets(batch):
    return {
        "query": [b["query"] for b in batch],
        "pos": [b["pos"] for b in batch],
        "neg": [b["neg"] for b in batch],
        "t_pos": torch.tensor([b["t_pos"] for b in batch], dtype=torch.float32),
        "t_neg": torch.tensor([b["t_neg"] for b in batch], dtype=torch.float32),
    }


def _record_to_triplet(record, rng):
    """Build one (query, pos, neg) triplet dict from a MS MARCO record.
    Returns None if the record has no positive or no negative passage.
    Teacher scores are placeholders (0.0), filled in later."""
    passages = record["passages"]["passage_text"]
    selected = record["passages"]["is_selected"]
    pos_idx = [i for i, s in enumerate(selected) if s == 1]
    neg_idx = [i for i, s in enumerate(selected) if s == 0]
    if not pos_idx or not neg_idx:
        return None
    return {
        "query": record["query"],
        "pos": passages[rng.choice(pos_idx)],
        "neg": passages[rng.choice(neg_idx)],
        "t_pos": 0.0,
        "t_neg": 0.0,
    }


def build_eval_set(triplets):
    """Group triplets by query into eval examples {query, docs, labels},
    label 1 for positive passages, 0 for negatives (deduplicated; positive wins)."""
    by_query = {}
    for t in triplets:
        docs = by_query.setdefault(t["query"], {})
        docs[t["pos"]] = max(docs.get(t["pos"], 0), 1)
        docs[t["neg"]] = max(docs.get(t["neg"], 0), 0)
    return [
        {"query": q, "docs": list(d.keys()), "labels": list(d.values())}
        for q, d in by_query.items()
    ]


def build_msmarco_triplets(data_cfg, seed=42):
    """Build (query, pos, neg) triplets from MS MARCO; teacher scores set later."""
    from datasets import load_dataset
    rng = random.Random(seed)
    ds = load_dataset(data_cfg.dataset_name, data_cfg.dataset_config, split="train")
    ds = ds.select(range(min(data_cfg.subset_size, len(ds))))
    triplets = []
    for rec in ds:
        t = _record_to_triplet(rec, rng)
        if t is not None:
            triplets.append(t)
    return triplets
