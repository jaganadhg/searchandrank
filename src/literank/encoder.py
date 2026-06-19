import torch
import torch.nn as nn
import torch.nn.functional as F


class DualEncoder(nn.Module):
    def __init__(self, cfg, encoder=None, tokenizer=None):
        super().__init__()
        self.cfg = cfg
        if tokenizer is None or encoder is None:
            from transformers import AutoModel, AutoTokenizer
            tokenizer = tokenizer or AutoTokenizer.from_pretrained(cfg.encoder_name)
            encoder = encoder or AutoModel.from_pretrained(cfg.encoder_name)
        self.tokenizer = tokenizer
        self.encoder = encoder
        if cfg.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.proj = nn.Linear(cfg.embed_dim, cfg.proj_dim) if cfg.proj_dim != cfg.embed_dim else None

    def tokenize(self, texts, max_len):
        return self.tokenizer(texts, padding="max_length", truncation=True,
                              max_length=max_len, return_tensors="pt")

    def embed(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        emb = out.last_hidden_state
        if self.proj is not None:
            emb = self.proj(emb)
        if self.cfg.l2_normalize:
            emb = F.normalize(emb, dim=-1)
        emb = emb * attention_mask.unsqueeze(-1).to(emb.dtype)
        return emb, attention_mask

    def encode(self, texts, max_len):
        enc = self.tokenize(texts, max_len)
        device = next(self.parameters()).device
        return self.embed(enc["input_ids"].to(device), enc["attention_mask"].to(device))
