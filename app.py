import random
import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

import sys

# Load data
responses_dict = joblib.load("Data/responses.joblib")
tags = joblib.load('Data/tags.joblib')

# Load the pretrained model
model_path = "model/"
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(tags))
model.eval()

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)


# Home page
@app.route('/')
def home():
    return render_template('home.html')


# Define event handler for socketIO
@socketio.on('connect')
def handle_connect():
    emit('response', {'data': 'Hello, welcome to ChatBot!'})


@socketio.on('input message')
def handle_input_message(message):
    # Get input sentences
    sentence = message['data']

    # Get the output
    encoding = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    output = model(input_ids, attention_mask=attention_mask)

    # Get the tag prediction
    prediction = torch.argmax(output.logits, dim=-1)
    tag = tags[prediction.item()]

    # Calculate probs to check how sure the model on the prediction
    probs = torch.softmax(output.logits, dim=-1)
    prob = probs[0, prediction.item()]

    # Check if the model is sure
    if prob.item() > 0.50:
        response = random.choice(responses_dict[tag])
        emit('response', {'data': response})
        if tag == 'bye':
            emit('disconnect request', {'data': 'disconnect'})
    else:
        response = "Sorry, I could not understand you. Please try again with different words."
        emit('response', {'data': response})


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=80, debug=True)
