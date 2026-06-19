import os
import torch


def save_embeddings(embs, masks, path):
    torch.save({"emb": embs.cpu().contiguous(), "mask": masks.cpu().contiguous()}, path)
    return os.path.getsize(path)


def load_embeddings(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["emb"], data["mask"]


def encode_and_cache(encoder, docs, max_len, path, batch_size=32):
    encoder.eval()
    embs, masks = [], []
    with torch.no_grad():
        for i in range(0, len(docs), batch_size):
            e, m = encoder.encode(docs[i:i + batch_size], max_len)
            embs.append(e.cpu())
            masks.append(m.cpu())
    return save_embeddings(torch.cat(embs), torch.cat(masks), path)
