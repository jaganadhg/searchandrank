#!/usr/bin/env python

import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertModel, AdamW
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import ndcg_score


class LITERanke():
    """
    A roush imliementation of the LITE-Rank model discussed in the paper.
    Efficient Document Ranking with Learnable Late Interactions
    https://arxiv.org/abs/2406.17968
    """
    def __init__(self, bert_model_name: str, mlp_w1: int, mlp_w2 : int) --> None:
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.mlp1 = nn.Sequential(
            nn.Linear(200, mlp_w1),  # Assuming document length of 200
            nn.ReLU(),
            nn.LayerNorm(mlp_w1),
            nn.Linear(mlp_w1, 1), 
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(30, mlp_w2),  # Assuming query length of 30
            nn.ReLU(),
            nn.LayerNorm(mlp_w2),
            nn.Linear(mlp_w2, 1)
        )
    

    def forward(self, query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask):
        # Obtain query and document embeddings from BERT
        query_output = self.bert_model(input_ids=query_input_ids, attention_mask=query_attention_mask)
        query_embeddings = query_output.last_hidden_state  # Q ∈ R^(P×L1)
        doc_output = self.bert_model(input_ids=doc_input_ids, attention_mask=doc_attention_mask)
        doc_embeddings = doc_output.last_hidden_state  # D ∈ R^(P×L2)

        # Calculate similarity matrix
        similarity_matrix = torch.matmul(query_embeddings.transpose(1, 2), doc_embeddings)  # S ∈ R^(L1×L2)

        # Apply separable LITE scorer
        row_updated = self.mlp1(similarity_matrix.transpose(1, 2)).squeeze(-1)  # Apply MLP1 row-wise
        score = self.mlp2(row_updated.transpose(1, 2)).squeeze(-1)  # Apply MLP2 column-wise 

        return score


class LITERankDataset(Dataset):
    def __init__(self, queries, documents, labels, tokenizer, max_query_len, max_doc_len):
        self.queries = queries
        self.documents = documents
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        document = self.documents[idx]
        label = self.labels[idx]

        query_inputs = self.tokenizer(query, 
                                      add_special_tokens=True, 
                                      max_length=self.max_query_len, 
                                      padding='max_length', 
                                      truncation=True, 
                                      return_tensors='pt')
        doc_inputs = self.tokenizer(document, 
                                    add_special_tokens=True, 
                                    max_length=self.max_doc_len, 
                                    padding='max_length', 
                                    truncation=True, 
                                    return_tensors='pt')
        return {
            'query_input_ids': query_inputs['input_ids'].squeeze(),
            'query_attention_mask': query_inputs['attention_mask'].squeeze(),
            'doc_input_ids': doc_inputs['input_ids'].squeeze(),
            'doc_attention_mask': doc_inputs['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.float)
        }




model_name = 'bert-base-uncased'
mlp1_width = 360
mlp2_width = 2400
max_query_len = 30
max_doc_len = 200
batch_size = 16
learning_rate = 2e-5
epochs = 3

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create datasets and dataloaders
train_dataset = RankingDataset(queries, documents, labels, tokenizer, max_query_len, max_doc_len)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize models
lite_model = SeparableLITE(model_name, mlp1_width, mlp2_width)
# Load your CE teacher model (implementation not provided here)
teacher_model = load_teacher_model() 

# Optimizer and loss function
optimizer = AdamW(lite_model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()  # You can experiment with KL loss as well

# Training loop
for epoch in range(epochs):
    for batch in train_dataloader:
        optimizer.zero_grad()

        # Get teacher scores
        with torch.no_grad():
            teacher_scores = teacher_model(batch['query_input_ids'], 
                                            batch['query_attention_mask'], 
                                            batch['doc_input_ids'], 
                                            batch['doc_attention_mask']) 

        student_scores = lite_model(batch['query_input_ids'], 
                                    batch['query_attention_mask'], 
                                    batch['doc_input_ids'], 
                                    batch['doc_attention_mask'])

        loss = loss_fn(student_scores, teacher_scores)
        loss.backward()
        optimizer.step()


