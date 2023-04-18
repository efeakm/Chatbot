# Chatbot
This is a chatbot application built using Flask and Huggingface's transformers library. The chatbot is able to take text input from a user and classify it into different tags, based on which it provides appropriate responses.

## Requirements
docker

## Usage
To use the Chatbot, follow the below steps:

1) Clone the repository
2) Run the following commands
    ```shell
    docker build -t chatbot .
    docker run -p 80:80 chatbot
    ```
3) Go to http://localhost:80/ in your browser to access the chatbot interface.

## Code
The app.py file contains the code for the Flask application and the socketIO event handlers. The code loads the pre-trained DistilBERT model and the corresponding tokenizer, along with the response data and tags. The handle_input_message() function handles the incoming message from the user, classifies it using the model, and returns an appropriate response based on the predicted tag.

The application runs on the Flask development server and can be accessed on port 80 by default. It uses SocketIO to enable real-time communication between the server and the client.

License
This project is licensed under the MIT License.
