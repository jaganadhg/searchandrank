#!/usr/bin/env python

import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertModel, AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers import BertTokenizer
from datasets import load_dataset
from sklearn.metrics import ndcg_score


from encutils import extract_embedding, load_teacher_model


class LITERank(nn.Module):
    """
    A roush imliementation of the LITE-Rank model discussed in the paper.
    Efficient Document Ranking with Learnable Late Interactions
    https://arxiv.org/abs/2406.17968
    """

    def __init__(self, transf_dim: int, mlp1_dim: int, mlp2_dim: int) -> None:
        """
        transf_dim = 768  # BERT-base embedding dimension
        mlp1_dim = 512    # Hidden dimension for the first MLP
        mlp2_dim = 256    # Hidden dimension for the second MLP

        lite_rank_model = LITERank(transf_dim, mlp1_dim, mlp2_dim)
        """
        super(LITERank, self).__init__()
        self.row_mlp = nn.Sequential(
            nn.Linear(transf_dim, mlp1_dim),
            nn.ReLU(),
            nn.LayerNorm(mlp1_dim),
            nn.Linear(mlp1_dim, transf_dim),
            nn.ReLU(),
            nn.LayerNorm(transf_dim),
        )
        self.col_mlp = nn.Sequential(
            nn.Linear(transf_dim, mlp2_dim),
            nn.ReLU(),
            nn.LayerNorm(mlp2_dim),
            nn.Linear(mlp2_dim, transf_dim),
            nn.ReLU(),
            nn.LayerNorm(transf_dim),
        )

        self.final_projection = nn.Linear(transf_dim, 1)

    def forward(self, query_emb: torch.Tensor, doc_emb: torch.Tensor) -> torch.Tensor:
        """
        query_emb: [batch_size, query_length, embedding_dim]
        doc_emb: [batch_size, doc_length, embedding_dim]
        Einsum Notation:
            "bqd,brd->bqr"
            b: Batch size
            q: Query length
            r: Document length
            d: Embedding dimension
        """
        sim_mat = torch.einsum("bqd,brd->bqr", query_emb, doc_emb)

        # Apply MLPs (note the order: rows then columns)
        interm_mat = self.row_mlp(sim_mat)
        final_matrix = self.col_mlp(interm_mat.transpose(1, 2)).transpose(1, 2)

        # Project to a scalar score
        scores = self.final_projection(final_matrix).squeeze(-1)

        return scores


class LITERankDataset(Dataset):

    def __init__(
        self,
        dataset_name: str = "microsoft/ms_marco",
        tokenizer_name: str = "bert-base-uncased",
        split: str = "train",
    ):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.dataset = load_dataset(dataset_name, "v2.1", split=split)
        self.dataset = self.dataset.select(range(24))
        self.processed_data = self._process_data()

    def _process_data(self):
        processed_data = []
        for record in self.dataset:
            #print(record)
            query = record["query"]
            encoded_q = extract_embedding(query)
            # passages = record["passages"]
            passages = record["passages"]["passage_text"]
            selcted_lable = record["passages"]["is_selected"]
            for idx, passage in enumerate(passages):
                encoded_p = extract_embedding(passage)
                data_local = {
                    "label": selcted_lable[idx],
                    "query": query,
                    "passage": passage,
                    "query_enc": encoded_q,
                    "passage_enc": encoded_p,
                }
                processed_data.append(data_local)

                # processed_data.append((query, passage["passage_text"]))
        return processed_data

    def __getitem__(self, idx):
        return self.processed_data[idx]

    def __len__(self):
        return len(self.processed_data)


if __name__ == "__main__":

    transf_dim = 768  # BERT-base embedding dimension
    mlp1_dim = 512  # Hidden dimension for the first MLP
    mlp2_dim = 256  # Hidden dimension for the second MLP

    batch_size = 16
    learning_rate = 2e-5
    epochs = 3

    lite_rank_model = LITERank(transf_dim, mlp1_dim, mlp2_dim)

    print("Lite Rank data loader")
    train_dataset = LITERankDataset(
        dataset_name="microsoft/ms_marco",
        tokenizer_name="bert-base-uncased",
        split="train",
    )
    print("Torch Data Loader")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("Teacher Model")
    teacher_model = load_teacher_model("cross-encoder/ms-marco-MiniLM-L-6-v2")

    optimizer = AdamW(lite_rank_model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    print("Training loop")

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        for batch in train_dataloader:
            print(batch)
            optimizer.zero_grad()

            # Get teacher scores
            with torch.no_grad():
                teacher_scores = teacher_model(batch["query"], batch["passage"])

            student_scores = lite_rank_model(batch["query_enc"], batch["passage_enc"])

            loss = loss_fn(student_scores, teacher_scores)
            loss.backward()
            optimizer.step()

# model_name = 'bert-base-uncased'
# mlp1_width = 360
# mlp2_width = 2400
# max_query_len = 30
# max_doc_len = 200
# batch_size = 16
# learning_rate = 2e-5
# epochs = 3

# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# dataset = load_dataset("coco_marc")

# queries = dataset["train"]["query"]
# documents = dataset["train"]["document"]
# labels = dataset["train"]["label"]

# # Create the dataset and dataloader
# max_query_len = 30
# max_doc_len = 200
# batch_size = 16


# # Create datasets and dataloaders
# train_dataset = LITERankDataset(
#     queries, documents, labels, tokenizer, max_query_len, max_doc_len
# )
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# # Initialize models
# lite_model = LITERanke(model_name, mlp1_width, mlp2_width)


# def load_teacher_model(model_name: str):
#     """
#     Load the teacher model from the Hugging Face model hub
#     """
#     teacher_model = BertModel.from_pretrained(model_name)
#     return teacher_model


# teacher_model = load_teacher_model()

# # Optimizer and loss function
# optimizer = AdamW(lite_model.parameters(), lr=learning_rate)
# loss_fn = nn.MSELoss()  # You can experiment with KL loss as well

# # Training loop
# for epoch in range(epochs):
#     for batch in train_dataloader:
#         optimizer.zero_grad()

#         # Get teacher scores
#         with torch.no_grad():
#             teacher_scores = teacher_model(batch['query_input_ids'],
#                                             batch['query_attention_mask'],
#                                             batch['doc_input_ids'],
#                                             batch['doc_attention_mask'])

#         student_scores = lite_model(batch['query_input_ids'],
#                                     batch['query_attention_mask'],
#                                     batch['doc_input_ids'],
#                                     batch['doc_attention_mask'])

#         loss = loss_fn(student_scores, teacher_scores)
#         loss.backward()
#         optimizer.step()
