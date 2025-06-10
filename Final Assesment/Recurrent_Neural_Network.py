import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Generate Time Series Data (Sine Wave)
def generate_sine_wave(seq_len, total_samples):
    x = np.linspace(0, total_samples, total_samples)
    data = np.sin(x)
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

SEQ_LEN = 50
TOTAL_SAMPLES = 1000

X, y = generate_sine_wave(SEQ_LEN, TOTAL_SAMPLES)
X = X.reshape(-1, SEQ_LEN, 1) # (samples, sequence_length, input_size)
y = y.reshape(-1, 1) # (samples, output_size)

# Prepare DataLoader
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define the RNN Model
class RNNPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNPredictor, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x) # out: [batch, seq_len, hidden]
        out = out[:, -1, :] # last time step
        out = self.fc(out)
        return out
    
model = RNNPredictor(input_size=1, hidden_size=50, output_size=1)

# Train the Model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

# Evaluate and Plot Results
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for xb, yb in test_loader:
        pred = model(xb)
        predictions.extend(pred.numpy())
        actuals.extend(yb.numpy())

plt.plot(actuals, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.title("Time Series Prediction (PyTorch RNN)")
plt.show()
