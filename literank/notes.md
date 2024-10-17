from datasets import load_dataset
dataset = load_dataset("ms_marco", "v2.1")
small_dataset = dataset['train'].select(range(2))
queries = small_dataset['query']
documents = small_dataset['passage']
labels = small_dataset['label']
small_dataset
queries = small_dataset['query']
documents = small_dataset['passages']
labels = small_dataset['label']
small_dataset
small_dataset[0]
small_dataset = dataset['train'].select(range(10))
queries = small_dataset['query']
passages = [p['passage_text'] for p in small_dataset['passages']]
labels = [1 if p['is_selected'] else 0 for p in small_dataset['passages']]
len(queries)
len(passages)
passages[0]
labelms
labels
sample_data[0]
small_data[0]
small_dataset[0]
small_dataset[0]['passage']
small_dataset[0]['passages']
small_dataset[0]['passages']['passage_text']
small_dataset[0]['passages']['passage_text'][0]
from transformers import BertTokenizer
q = small_dataset[0]['query']
p = small_dataset[0]['passages']['passage_text'][0]
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
ec = tokenizer(q, p, padding='max_length', truncation=True, return_tensors="pt")
ec
ec[0]
type(ec)
ec = tokenizer(q, padding='max_length', truncation=True, return_tensors="pt")
ec
ec = tokenizer(p, padding='max_length', truncation=True, return_tensors="pt")
ec
bert_model = BertModel.from_pretrained("bert-base-uncased")
from transformers import BertModel, BertTokenizer
bert_model = BertModel.from_pretrained("bert-base-uncased")
encoded_inputs = tokenizer(q, p, return_tensors="pt", padding=True, truncation=True)
input_ids = encoded_inputs["input_ids"]
attention_mask = encoded_inputs["attention_mask"]


### Create CE Data Sample
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

# Load the cross-encoder model
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Prepare the training data
# Example data: list of InputExample objects with (query, document, label)
train_data = [
    InputExample(texts=["query1", "document1"], label=1),
    InputExample(texts=["query2", "document2"], label=0),
    # Add more training examples here
]

# Create a DataLoader
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)

# Define the training parameters
num_epochs = 1
warmup_steps = 100
output_path = "output/cross-encoder-ms-marco-MiniLM-L-6-v2"

# Train the model
model.fit(train_dataloader=train_dataloader,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=output_path)