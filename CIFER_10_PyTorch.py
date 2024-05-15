import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torchvision.utils as vutils
import os
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Discriminator
def define_discriminator(in_channels=3):
    model = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Flatten(),
        nn.Dropout(0.4),
        nn.Linear(256*4*4, 1),
        nn.Sigmoid()
    )
    return model

# Generator
def define_generator(latent_dim):
    model = nn.Sequential(
        nn.Linear(latent_dim, 256*4*4),
        nn.LeakyReLU(0.2, inplace=True),
        Unflatten(),
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
        nn.Tanh()
    )
    return model

# Unflatten Layer (helper for Generator)
class Unflatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 256, 4, 4)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


# Function to save generated images
def save_generated_images(fake_images, epoch, folder="G-images"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, f"epoch_{epoch+1}.png")
    with torch.no_grad():
        vutils.save_image(fake_images.detach(), filename, normalize=True)

# Function to save model
def save_model(model, epoch, folder="savedModels", model_name="generator"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(model.state_dict(), os.path.join(folder, f"{model_name}_epoch_{epoch+1}.pt"))

# Function to evaluate and summarize performance
def summarize_performance(epoch, g_model, d_model, dataloader, device, latent_dim):
    g_model.eval()
    d_model.eval()
    
    real_correct = 0
    fake_correct = 0
    total = 0
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        total += batch_size

        # Evaluate discriminator on real images
        real_labels = torch.ones(batch_size, 1).to(device)
        real_outputs = d_model(real_images)
        real_correct += (real_outputs > 0.5).sum().item()

        # Evaluate discriminator on fake images
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = g_model(z)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        fake_outputs = d_model(fake_images)
        fake_correct += (fake_outputs < 0.5).sum().item()

        if i == 0:  # Save images from the first batch
            save_generated_images(fake_images, epoch)

    real_accuracy = 100 * real_correct / total
    fake_accuracy = 100 * fake_correct / total
    print(f"Epoch {epoch+1}: Discriminator Real Accuracy: {real_accuracy:.2f}%, Fake Accuracy: {fake_accuracy:.2f}%")

# Training Function
def train(g_model, d_model, dataloader, latent_dim, n_epochs=200, device=device):
    for epoch in range(n_epochs):
        g_model.train()
        d_model.train()

        for i, (real_images, _) in enumerate(dataloader):
            # Training Discriminator
            d_model.zero_grad()
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            outputs = d_model(real_images)
            d_loss_real = F.binary_cross_entropy(outputs, real_labels)
            d_loss_real.backward()

            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = g_model(z)
            outputs = d_model(fake_images)
            d_loss_fake = F.binary_cross_entropy(outputs, fake_labels)
            d_loss_fake.backward()
            d_optimizer.step()

            # Training Generator
            g_model.zero_grad()
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = g_model(z)
            outputs = d_model(fake_images)
            g_loss = F.binary_cross_entropy(outputs, real_labels)
            g_loss.backward()
            g_optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(dataloader)}], d_loss: {d_loss_real.item() + d_loss_fake.item():.4f}, g_loss: {g_loss.item():.4f}')

        # Evaluate and save models periodically
        if (epoch+1) % 10 == 0:
            summarize_performance(epoch, g_model, d_model, dataloader, device, latent_dim)
            save_model(g_model, epoch, model_name="generator")
            save_model(d_model, epoch, model_name="discriminator")


# Hyperparameters
latent_dim = 100
d_model = define_discriminator().to(device)
g_model = define_generator(latent_dim).to(device)

# Optimizers
d_optimizer = optim.Adam(d_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = optim.Adam(g_model.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Train the models
train(g_model, d_model, train_loader, latent_dim)
