import os
import json
import random
import re
import string
import datetime
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def simple_tokenize(text):
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    return text.split()


class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class ChatbotAssistant:
    def __init__(self, intents_path, function_mappings=None):
        self.model = None
        self.intents_path = intents_path
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        self.function_mappings = function_mappings or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X = None
        self.y = None

    def tokenize_text(self, text):
        return simple_tokenize(text)

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        if not os.path.exists(self.intents_path):
            print(f"Error: Intents file not found at {self.intents_path}")
            exit(1)

        with open(self.intents_path, 'r') as f:
            intents_data = json.load(f)

        for intent in intents_data['intents']:
            tag = intent['tag']
            if tag not in self.intents:
                self.intents.append(tag)
                self.intents_responses[tag] = intent['responses']
            for pattern in intent['patterns']:
                tokens = self.tokenize_text(pattern)
                self.vocabulary.extend(tokens)
                self.documents.append((tokens, tag))

        self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags, indices = [], []
        for words, tag in self.documents:
            bags.append(self.bag_of_words(words))
            indices.append(self.intents.index(tag))
        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size=8, lr=0.001, epochs=100):
        X_tensor = torch.tensor(self.X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(self.y, dtype=torch.long).to(self.device)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)

        self.model = ChatbotModel(self.X.shape[1], len(self.intents)).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(loader):.4f}")

    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)
        with open(dimensions_path, 'w') as f:
            json.dump({
                'input_size': self.X.shape[1],
                'output_size': len(self.intents),
                'vocabulary': self.vocabulary,
                'intents': self.intents,
                'responses': self.intents_responses
            }, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            data = json.load(f)

        self.vocabulary = data['vocabulary']
        self.intents = data['intents']
        self.intents_responses = data['responses']
        self.model = ChatbotModel(data['input_size'], data['output_size']).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def process_message(self, message):
        if not message.strip():
            return "Please say something."
        words = self.tokenize_text(message)
        bag = self.bag_of_words(words)
        input_tensor = torch.tensor([bag], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted_index = torch.max(probs, dim=1)

        confidence_value = confidence.item()
        predicted_intent = self.intents[predicted_index.item()]

        if confidence_value < 0.6:
            return "I'm not sure I understand. Can you rephrase?"

        if predicted_intent in self.function_mappings:
            worker_response = self.function_mappings[predicted_intent]()
            return worker_response

        return random.choice(self.intents_responses.get(predicted_intent, ["I'm not sure how to help with that."]))


def get_stocks():
    stocks = ['AAPL', 'META', 'NVDA', 'GOOG', 'MSFT']
    selected = random.sample(stocks, 3)
    return f"Your top stocks today: {', '.join(selected)}"

def get_time():
    return f"The current time is {datetime.datetime.now().strftime('%H:%M:%S')}."

def tell_joke():
    jokes = [
        "Why did the programmer quit his job? Because he didn't get arrays!",
        "What do you call 8 hobbits? A hobbyte.",
        "Why do Java developers wear glasses? Because they can't C#!"
    ]
    return random.choice(jokes)


if __name__ == '__main__':
    assistant = ChatbotAssistant(
        intents_path='intents.json',
        function_mappings={
            'stocks': get_stocks,
            'time': get_time,
            'joke': tell_joke
        }
    )

    assistant.parse_intents()
    assistant.prepare_data()

    if len(assistant.documents) == 0:
        print("Error: No training data found.")
        exit(1)

    print(f"Training model on {len(assistant.documents)} patterns.")
    assistant.train_model(epochs=100)
    assistant.save_model('chatbot_model.pth', 'chatbot_dims.json')

    print("\n--- Chatbot READY ---")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == '/quit':
            print("Bot: Goodbye!")
            break
        response = assistant.process_message(user_input)
        print(f"Bot: {response}")
