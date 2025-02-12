import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: XOR Data
data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([[0], [1], [1], [0]])

# Convert to tensors
X = torch.tensor(data, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32)

# Step 2: Define the MLP Model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation="relu"):
        super(MLP, self).__init__()
        
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError("Unsupported activation function")
        
    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = torch.sigmoid(self.output(x))  # Sigmoid output for binary classification
        return x

# Step 3: Initialize Model, Loss, and Optimizer
model = MLP(input_size=2, hidden_size=4, output_size=1, activation="relu")
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for XOR
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Step 4: Train the Model
num_epochs = 5000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 500 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Step 5: Test the Model
with torch.no_grad():
    predictions = model(X).round()
    print("\nPredictions:")
    print(predictions.numpy())
