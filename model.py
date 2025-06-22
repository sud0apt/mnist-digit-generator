import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=20, label_dim=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_dim = label_dim

        self.fc1 = nn.Linear(784 + label_dim, 512)
        self.fc_hidden_enc = nn.Linear(512, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)


        self.fc3 = nn.Linear(latent_dim + label_dim, 400)
        self.fc_hidden_dec = nn.Linear(400, 512)
        self.fc4 = nn.Linear(512, 784)


    def encode(self, x, y):
        x = torch.cat([x, y], dim=1)
        h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.fc_hidden_enc(h1))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        z = torch.cat([z, y], dim=1)
        h3 = F.relu(self.fc3(z))
        h3 = F.relu(self.fc_hidden_dec(h3))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar
