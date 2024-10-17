#!usrbin/env python

import torch
from transformers import BertModel, BertTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")


def encode_text(text: str) -> dict:
    """
    Tokenize a single text using BERT tokenizer.

    Args:
    text: The text to tokenize.

    Returns:
    encoded_inputs: Tokenized outputs including input_ids and attention_mask.
    """
    encoded_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    return encoded_inputs


def extract_embedding(text: str) -> torch.Tensor:
    """
    Extract embeddings for a single text using BERT model.

    Args:
    text: The text to extract embeddings for.

    Returns:
    embedding: Embeddings for the text.
    """

    encoded_inputs = encode_text(text)
    input_ids = encoded_inputs["input_ids"]
    attention_mask = encoded_inputs["attention_mask"]

    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

    # Use the [CLS] token representation as the embedding
    embedding = last_hidden_state[:, 0, :]
    return embedding


def extract_query_and_document_embeddings(query: str, document: str) -> tuple:
    """
    Extract query and document embeddings independently using BERT model.

    Args:
    query: The query text.
    document: The document text.

    Returns:
    tuple: Query and document embeddings.
    """
    query_emb = extract_embedding(query)
    doc_emb = extract_embedding(document)
    return query_emb, doc_emb


def load_teacher_model(model_name: str):
    """
    Load the teacher model from the Hugging Face model hub
    """
    teacher_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return teacher_model


# Example usage
if __name__ == "__main__":
    query = "What is the capital of France?"
    document = "Paris is the capital and most populous city of France."
    query_emb, doc_emb = extract_query_and_document_embeddings(query, document)
    print("Query Embedding:", query_emb)
    print("Document Embedding:", doc_emb)
