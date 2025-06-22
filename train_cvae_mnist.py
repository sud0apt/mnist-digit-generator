import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import ConditionalVAE
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss function
def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

# Data
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# One-hot encoding
def one_hot(labels, num_classes=10):
    return torch.eye(num_classes)[labels]

# Train
model = ConditionalVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 50

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for x, labels in train_loader:
        x = x.view(-1, 784).to(device)
        y = one_hot(labels).to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(x, y)
        loss = loss_function(recon_batch, x, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset):.4f}")

# Save model
torch.save(model.state_dict(), "vae_mnist.pth")
print("✅ Model saved to vae_mnist.pth")
