# 🤖 AI Chatbot Assistant

A simple AI chatbot built using Python and PyTorch. It uses a neural network to classify user inputs and respond based on a trained intents dataset. You can also extend it with custom functional responses like fetching stocks, telling jokes, and more.

## 🚀 Features

- 🔍 Intent classification using a feed-forward neural network
- 🧠 Trainable on a custom `intents.json` file
- 🗣️ Responds to greetings, programming questions, and more
- ⚙️ Extendable with custom Python functions (`get_stocks`, `tell_joke`, `get_time`, etc.)
- 📁 Save/load trained models and vocabulary for later use
- 🧵 Modular design and easy to customize

## 📂 Project Structure

```
chatbot/
│
├── chatbot.py               # Main chatbot logic & training
├── intents.json             # Intents file (tags, patterns, responses)
├── chatbot_model.pth        # Trained model weights
├── chatbot_dims.json        # Model dimensions and vocab info
├── README.md                # This file
```

## 🧠 How It Works

1. Loads the `intents.json` file containing tags, patterns, and responses.
2. Tokenizes and vectorizes the input using a simple bag-of-words method.
3. Trains a neural network model to classify input messages into intent tags.
4. Responds with a matching response or calls a mapped Python function.

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/chatbot-assistant.git
   cd chatbot-assistant
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   If no `requirements.txt` yet, here’s what you need:
   ```bash
   pip install torch numpy
   ```

## 🛠️ Training the Bot

Run this to train and save the model:

```bash
python chatbot.py
```

On first run, it will train using `intents.json` and save the model to disk.

## 💬 Using the Chatbot

After training:

```bash
python chatbot.py
```

Type a message like:

```
You: Hello
Bot: Hi there, how can I assist you today?
```

Type `/quit` to exit.

## 🧩 Add Your Own Functions

You can map custom actions to tags using `function_mappings`:

```python
assistant = ChatbotAssistant(
    intents_path='intents.json',
    function_mappings={
        'stocks': get_stocks,
        'joke': tell_joke,
        'time': get_time
    }
)
```

These functions can do anything you want — return a string, call APIs, etc.

## 📄 Example Intents

```json
{
  "tag": "greeting",
  "patterns": ["Hi", "Hello", "Hey"],
  "responses": ["Hello!", "Hi there!", "How can I help you?"]
}
```

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or pull requests with improvements.

## 📜 License

MIT License. Feel free to use and modify.

## 👨‍💻 Author

Created by [Your Name]. Feel free to reach out for collaborations!
