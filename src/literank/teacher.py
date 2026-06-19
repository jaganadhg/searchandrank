import json
import os


class CrossEncoderTeacher:
    def __init__(self, name, device="cpu", max_len=256):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        self.device = device
        self.max_len = max_len

    def score(self, queries, passages, batch_size=32):
        out = []
        for i in range(0, len(queries), batch_size):
            q = queries[i:i + batch_size]
            p = passages[i:i + batch_size]
            enc = self.tokenizer(q, p, padding=True, truncation=True,
                                 max_length=self.max_len, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                logits = self.model(**enc).logits.squeeze(-1)
            out.extend(logits.detach().cpu().tolist())
        return out


def add_teacher_scores(triplets, teacher, batch_size=32):
    queries = [t["query"] for t in triplets]
    pos = [t["pos"] for t in triplets]
    neg = [t["neg"] for t in triplets]
    t_pos = teacher.score(queries, pos, batch_size=batch_size)
    t_neg = teacher.score(queries, neg, batch_size=batch_size)
    for t, sp, sn in zip(triplets, t_pos, t_neg):
        t["t_pos"], t["t_neg"] = float(sp), float(sn)
    return triplets


def cache_teacher_scores(triplets, teacher, path, batch_size=32):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    out = add_teacher_scores(triplets, teacher, batch_size=batch_size)
    with open(path, "w") as f:
        json.dump(out, f)
    return out
