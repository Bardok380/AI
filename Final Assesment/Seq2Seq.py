import torch
import torch.nn as nn
import torch.optim as optim
import random

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Toy dataset
pairs = [
    ["hi", "hello"],
    ["how are you", "i am fine"],
    ["what is your name", "i am a chatbot"],
    ["bye", "goodbye"]
]

# Tokenizer and Vocabulary
def tokenize(sentence):
    return sentence.lower().split()

# Buikd vocabulary
all_words = set()
for pair in pairs:
    all_words.update(tokenize(pair[0]))
    all_words.update(tokenize(pair[1]))

word2idx = {word: i+2 for i, word in enumerate(all_words)}
word2idx["<pad>"] = 0
word2idx["<eos>"] = 1
idx2word = {i: w for w, i in word2idx.items()}

vocab_size = len(word2idx)

# Convert sentences to tensor of word indices
def sentence_to_tensor(sentence):
    tokens = tokenize(sentence) + ["<eos>"]
    idxs = [word2idx[word] for word in tokens]
    return torch.tensor(idxs, dtype=torch.long, device=device).unsqueeze(1)

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell
    
# Decoder
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, cell):
        embedded = self.embedding(input).unsqueeze(0)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(0))
        return prediction, hidden, cell

# Seq2Seq wrapper
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder =encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        max_len = trg.size(0)
        batch_size = trg.size(1)
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(max_len, batch_size, vocab_size).to(device)
        hidden, cell = self.encoder(src)
        input =trg[0, :]

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            top1 = output.argmax(1)
            input = trg[t] if random.random() < teacher_forcing_ratio else top1
        return outputs

# Model setup
hidden_size = 256
encoder = Encoder(vocab_size, hidden_size).to(device)
decoder = Decoder(hidden_size, vocab_size).to(device)
model = Seq2Seq(encoder, decoder).to(device)

# Training setup
criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Prepare training data
pairs_tensors = [(sentence_to_tensor(q), sentence_to_tensor(a)) for q, a in pairs]

# Training Loop
for epoch in range(300):
    total_loss = 0
    for src, trg in pairs_tensors:
        optimizer.zero_grad()
        output = model(src, trg)
        loss = criterion(output[1:].reshape(-1, vocab_size), trg[1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# Chat function
def chat(input_sentence):
    with torch.no_grad():
        input_tensor = sentence_to_tensor(input_sentence)
        hidden, cell = encoder(input_tensor)

        input_token = torch.tensor([word2idx["<eos>"]], device=device)
        response = []
        for _ in range(10):
            output, hidden, cell = decoder(input_token, hidden, cell)
            top1 = output.argmax(1).item()
            if top1 == word2idx["<eos>"]:
                break
            response.append(idx2word[top1])
            input_token = torch.tensor([top1], device=device)
        return ' '.join(response)
    
# Test the chatbot
print("\nChatbot Test:")
while True:
    sentence = input("You: ")
    if sentence == "quit":
        break
    print("Bot:", chat(sentence))
