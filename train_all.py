import json
import numpy as np
import time
import joblib

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
from torch.utils.data import TensorDataset, DataLoader

# Load and combine data
with open('Data/intents.json', 'r') as f:
    intents = json.load(f)

tags = []
x, y = [], []
for intent in intents['intents']:
    tags.append(intent['tag'])
    for pattern in intent['patterns']:
        x.append(pattern)
        y.append(tags.index(intent['tag']))

# Load the model and the tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device =', device)

model_name = 'distilbert-base-uncased'
configuration = AutoConfig.from_pretrained(model_name)
configuration.hidden_dropout_prob = 0.8
configuration.attention_probs_dropout_prob = 0.8
configuration.num_labels = len(tags)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                           config=configuration).to(device)


# Create tensors
encoding = tokenizer(x, truncation=True, padding=True, return_tensors='pt')
input_ids, att_mask = encoding['input_ids'], encoding['attention_mask']
labels = torch.tensor(y, dtype=torch.int64)
dataset = TensorDataset(input_ids, att_mask, labels)

batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Train the model on the full data
learning_rate = 5e-5
num_epochs = 15
weight_decay = 0.05
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

start = time.time()
for epoch in range(num_epochs):

    # Train on the training data
    model.train()
    train_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids, att_mask, labels = (x.to(device) for x in batch)
        outputs = model(input_ids, attention_mask=att_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Calculate avg epoch loss over all batches
    train_loss = train_loss / len(dataloader)

    print(f"Epoch {epoch + 1}: Training Loss = {train_loss}")

print(f'elapsed time = {round(time.time() - start, 2)}')

# Save the model
path = 'model/'
model.save_pretrained(path)
joblib.dump(tags, 'Data/tags.joblib')
