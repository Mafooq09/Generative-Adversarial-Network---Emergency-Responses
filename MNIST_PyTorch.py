import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Define the discriminator model
def define_discriminator(in_shape=(1, 28, 28)):
    model = nn.Sequential(
        nn.Conv2d(in_shape[0], 64, 3, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.4),
        nn.Conv2d(64, 64, 3, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.4),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 1),
        nn.Sigmoid()
    )
    return model

# Define the generator model
def define_generator(latent_dim):
    model = nn.Sequential(
        nn.Linear(latent_dim, 128 * 7 * 7),
        nn.LeakyReLU(0.2),
        nn.Unflatten(1, (128, 7, 7)),
        nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(128, 1, 7, padding=3),
        nn.Sigmoid()
    )
    return model

# Function to load MNIST dataset
def load_real_samples():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return DataLoader(train_data, batch_size=64, shuffle=True)

# Generate real samples with labels
def generate_real_samples(dataloader, n_samples):
    # Select a random batch of images
    real_images, _ = next(iter(dataloader))
    return real_images[:n_samples], torch.ones((n_samples, 1))

# Generate latent points
def generate_latent_points(latent_dim, n_samples):
    return torch.randn(n_samples, latent_dim)

# Use the generator to generate n fake examples
def generate_fake_samples(generator, latent_dim, n_samples):
    latent_points = generate_latent_points(latent_dim, n_samples)
    fake_images = generator(latent_points)
    return fake_images, torch.zeros((n_samples, 1))

# Function for saving plots of generated images
def save_plot(examples, epoch, n=10):
    num_examples = examples.shape[0]
    grid_size = int(np.sqrt(num_examples))
    examples = examples.detach().cpu().numpy()
    plt.figure(figsize=(9, 9))
    for i in range(num_examples):
        plt.subplot(grid_size, grid_size, i+1)
        plt.imshow(examples[i, 0, :, :], cmap='gray_r')
        plt.axis('off')
    plt.savefig(f'generated_plot_e{epoch+1:03d}.png')
    plt.close()

# Train the generator and discriminator
def train(generator, discriminator, gan_model, dataloader, latent_dim, n_epochs=1, n_batch=256):
    for epoch in range(n_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            n_samples = real_images.size(0)

            # Train Discriminator
            discriminator.zero_grad()
            real_labels = torch.ones(n_samples, 1)
            fake_labels = torch.zeros(n_samples, 1)
            real_loss = nn.BCELoss()(discriminator(real_images), real_labels)
            fake_images = generate_fake_samples(generator, latent_dim, n_samples)[0]
            fake_loss = nn.BCELoss()(discriminator(fake_images.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            generator.zero_grad()
            g_loss = nn.BCELoss()(discriminator(fake_images), real_labels)
            g_loss.backward()
            g_optimizer.step()

            # Generate a smaller number of fake samples for plotting
            plot_num_samples = 32  # Adjust this as needed
            fake_images = generate_fake_samples(generator, latent_dim, plot_num_samples)[0]
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(dataloader)}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')

        # Evaluate and save models at intervals
        if (epoch+1) % 10 == 0:
            # Save generated images
            save_plot(fake_images, epoch)

            # Save the model
            torch.save(generator.state_dict(), f'generator_epoch_{epoch+1}.pth')

# Instantiate models, optimizers and start training
latent_dim = 100
generator = define_generator(latent_dim)
discriminator = define_discriminator()
gan_model = nn.Sequential(generator, discriminator)  # Just for structural reference

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

dataloader = load_real_samples()
train(generator, discriminator, gan_model, dataloader, latent_dim)
