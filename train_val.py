import json
import numpy as np
import time
import joblib

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
from torch.utils.data import TensorDataset, DataLoader


# Load and separate data
with open('Data/intents.json', 'r') as f:
    intents = json.load(f)

tags = []
x, y = [], []
for intent in intents['intents']:
    tags.append(intent['tag'])
    for pattern in intent['patterns']:
        x.append(pattern)
        y.append(tags.index(intent['tag']))

# Separate data into train and validation
np.random.seed(145)
indices = np.random.permutation(len(x))
train_size = int(len(indices) * 0.80)

x_train, x_val = np.asarray(x)[indices[:train_size]], np.asarray(x)[indices[train_size:]]
y_train, y_val = np.asarray(y)[indices[:train_size]], np.asarray(y)[indices[train_size:]]

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
train_encoding = tokenizer(list(x_train), truncation=True, padding=True, return_tensors='pt')
val_encoding = tokenizer(list(x_val), truncation=True, padding=True, return_tensors='pt')

train_input_ids, train_att_mask = train_encoding['input_ids'], train_encoding['attention_mask']
train_labels = torch.tensor(y_train, dtype=torch.int64)
train_dataset = TensorDataset(train_input_ids, train_att_mask, train_labels)

val_input_ids, val_att_mask = val_encoding['input_ids'], val_encoding['attention_mask']
val_labels = torch.tensor(y_val, dtype=torch.int64)
val_dataset = TensorDataset(val_input_ids, val_att_mask, val_labels)


batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Fine-tune the model
learning_rate = 5e-5
num_epochs = 20
weight_decay = 0.05
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

start = time.time()
for epoch in range(num_epochs):

    # Train on the training data
    model.train()
    train_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids, att_mask, labels = (x.to(device) for x in batch)
        outputs = model(input_ids, attention_mask=att_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Calculate avg epoch loss over all batches
    train_loss = train_loss / len(train_dataloader)

    # Calculate evaluation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, att_mask, labels = (x.to(device) for x in batch)
            outputs = model(input_ids, attention_mask=att_mask, labels=labels)
            val_loss += outputs.loss.item()

    # Avg loss
    val_loss = val_loss / len(val_dataloader)

    print(f"Epoch {epoch + 1}: Training Loss = {train_loss}, Validation Loss = {val_loss}")

print(f'final loss = {val_loss:.4f} -- elapsed time = {round(time.time() - start, 2)}')

# Save the model
path = 'model/'
model.save_pretrained(path)
joblib.dump(tags, 'Data/tags.joblib')

