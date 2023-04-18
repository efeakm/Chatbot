import socketio

# Create a SocketIO client instance
sio = socketio.Client()

# Define event handlers
@sio.event
def connect():
    print('Connected to server')

@sio.on('response')
def handle_response(data):
    print(f"Bot: {data['data']}")

# Connect to the Flask SocketIO server
sio.connect('http://localhost:80', wait_timeout=10)

# Send messages to the chatbot
while True:
    message = input("You: ")
    if message.lower() == 'bye':
        sio.emit('input message', {'data': message})
        break
    sio.emit('input message', {'data': message})
