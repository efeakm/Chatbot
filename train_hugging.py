import json
import numpy as np
import time
import joblib

import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig

# Separate x and y
with open('Data/intents.json', 'r') as f:
    intents = json.load(f)

tags = [intent['tag'] for intent in intents['intents']]
x = [pattern for intent in intents['intents'] for pattern in intent['patterns']]
y = [tags.index(intent['tag']) for intent in intents['intents'] for _ in intent['patterns']]
x = np.asarray(x)
y = np.asarray(y)

# Separate data into train and validation
np.random.seed(145)
indices = np.random.permutation(x.shape[0])
train_size = int(len(indices) * 0.80)

training_idx, val_idx = indices[:train_size], indices[train_size:]
x_train, x_val = x[training_idx], x[val_idx]
y_train, y_val = y[training_idx], y[val_idx]

# Load the model and the tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device =', device)

model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name, num_labels=len(tags))
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config).to(device)
# Encoders
train_encoding = tokenizer(x_train.tolist(), truncation=True, padding=True)
val_encoding = tokenizer(x_val.tolist(), truncation=True, padding=True)

train_input_ids = torch.tensor(train_encoding['input_ids'])
train_att_mask = torch.tensor(train_encoding['attention_mask'])
val_input_ids = torch.tensor(val_encoding['input_ids'])
val_att_mask = torch.tensor(val_encoding['attention_mask'])

train_labels = torch.tensor(y_train, dtype=torch.int64)
val_labels = torch.tensor(y_val, dtype=torch.int64)

# Convert the data to tensor and create DataLoader
batch_size = 4
train_dataset = TensorDataset(train_input_ids, train_att_mask, train_labels)
val_dataset = TensorDataset(val_input_ids, val_att_mask, val_labels)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Fine Tune the model using Trainer
learning_rate = 3e-5
num_epochs = 30

training_args = TrainingArguments(
    output_dir='test_trainer',
    evaluation_strategy='epoch'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

start = time.time()
trainer.train()
print('elapsed time =', int(time.time() - start))

# Save the model
path = 'model/'
model.save_pretrained(path)
joblib.dump(tags, 'Data/tags.joblib')
