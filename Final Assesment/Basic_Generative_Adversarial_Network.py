import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.utils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Hyperameters
latent_size = 64
image_size = 784 #28x28
hidden_size = 256
batch_size = 100
num_epochs = 5
learning_rate = 0.0002

# Data Loader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
mnist = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

# Generator network
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, image_size),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)
    
# Discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(image_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)
    
# Initialize models
G = Generator()
D = Discriminator()

# Loss and optimizers
criterion = nn.BCELoss()
optimizerD = optim.Adam(D.parameters(), lr=learning_rate)
optimizerG = optim.Adam(G.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(dataloader):
        batch_size_curr = images.size(0)
        images = images.view(batch_size_curr, -1)

        real_labels = torch.ones(batch_size_curr, 1)
        fake_labels = torch.zeros(batch_size_curr, 1)

        # Train Discriminator
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)

        z = torch.randn(batch_size_curr, latent_size)
        fake_images = G(z)
        outputs = D(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        optimizerD.zero_grad()
        d_loss.backward()
        optimizerD.step()

        # Train Generator
        z = torch.randn(batch_size_curr, latent_size)
        fake_images = G(z)
        outputs = D(fake_images)
        g_loss = criterion(outputs, real_labels)

        optimizerG.zero_grad()
        g_loss.backward()
        optimizerG.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}")

    # Save sample images every 10 epochs
    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            fake_images = G(torch.randn(16, latent_size)).reshape(-1, 1, 28, 28)
            grid = torchvision.utils.make_grid(fake_images, nrow=4, normalize=True)
            plt.imshow(grid.permute(1, 2, 0).cpu())
            plt.title(f"Epoch {epoch+1}")
            plt.axis('off')
            plt.show()
