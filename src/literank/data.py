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


def build_msmarco_triplets(data_cfg, seed=42):
    """Build (query, pos, neg) triplets from MS MARCO; teacher scores set later."""
    from datasets import load_dataset
    rng = random.Random(seed)
    ds = load_dataset(data_cfg.dataset_name, data_cfg.dataset_config, split="train")
    ds = ds.select(range(min(data_cfg.subset_size, len(ds))))
    triplets = []
    for rec in ds:
        passages = rec["passages"]["passage_text"]
        selected = rec["passages"]["is_selected"]
        pos_idx = [i for i, s in enumerate(selected) if s == 1]
        neg_idx = [i for i, s in enumerate(selected) if s == 0]
        if not pos_idx or not neg_idx:
            continue
        p = passages[rng.choice(pos_idx)]
        n = passages[rng.choice(neg_idx)]
        triplets.append({"query": rec["query"], "pos": p, "neg": n,
                         "t_pos": 0.0, "t_neg": 0.0})
    return triplets
