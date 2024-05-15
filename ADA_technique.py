import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

# Parameters
batch_size = 5
image_size = 128
learning_rate = 0.0002
epochs = 700
latent_dim = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset Class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if file.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Transform
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
])

# User prompt for dataset selection
dataset_choice = input("Please select from the following for training: \n1. Fire\n2. Collapsed Building\n3. Flooded Areas\n4. Traffic Incidents\n5. Normal\nEnter your choice (1-5): ")

# Map user choice to dataset directory
dataset_directories = {
    "1": "fire",
    "2": "collapsed_building",
    "3": "flooded_areas",
    "4": "traffic_incident",
    "5": "normal"
}

# Validate user input
if dataset_choice not in dataset_directories:
    raise ValueError("Invalid choice. Please select a number between 1 and 5.")

# Load the selected dataset
selected_dataset = dataset_directories[dataset_choice]
dataset_path = f'/home/shassa01/THESIS/AIDER/{selected_dataset}'
dataset = CustomDataset(root_dir=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

# Update save paths based on selected dataset
save_model_folder = f'home/shassa01/THESIS/Code/savedModels/{selected_dataset}'
save_image_folder = f'/home/shassa01/THESIS/Code/G-images/{selected_dataset}'

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input size: 3 x 128 x 128 (assuming RGB images)
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 64 x 64 x 64
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 128 x 32 x 32
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 256 x 16 x 16
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 512 x 8 x 8
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 1024 x 4 x 4
        )
        self.final = nn.Sequential(
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Final state size: 1 x 1 x 1
        )

    def forward(self, input):
        intermediate = self.main(input)
        output = self.final(intermediate)
        return output.view(-1)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is the latent vector Z, going into a convolution
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # State size: 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State size: 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State size: 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # State size: 64 x 32 x 32
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # State size: 32 x 64 x 64
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output state size: 3 x 128 x 128 (RGB image)
        )

    def forward(self, input):
        return self.main(input)


class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, z):
        # Generate an image from the noise vector z
        img = self.generator(z)
        # Evaluate the image with the discriminator
        validity = self.discriminator(img)
        return validity
		

def apply_augmentations(images, p):
    if torch.rand(1) < p:
        return augmentations(images)
    return images

def generate_latent_points(batch_size, latent_dim):
    z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
    return z


def generate_fake_samples(generator, batch_size, latent_dim):
    z = generate_latent_points(batch_size, latent_dim)
    fake_images = generator(z)
    fake_labels = torch.zeros(batch_size, device=device)
    return fake_images, fake_labels
	
def evaluate_performance(discriminator, dataloader, generator, device):
    discriminator.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            labels_real = torch.ones(images.size(0), device=device)
            preds_real = discriminator(images)
            correct += ((preds_real > 0.5) == labels_real).sum().item()
            
            z = generate_latent_points(images.size(0), latent_dim)
            fake_images = generator(z)
            labels_fake = torch.zeros(fake_images.size(0), device=device)
            preds_fake = discriminator(fake_images)
            correct += ((preds_fake < 0.5) == labels_fake).sum().item()
            
            total += labels_real.size(0) + labels_fake.size(0)
    
    discriminator.train()
    return correct / total

# Function to get real samples
def get_real_samples(batch_size):
    real_images = next(iter(dataloader))
    real_labels = torch.ones(batch_size, device=device)
    return real_images.to(device), real_labels
	
	
# Saving Generated Images
def save_generated_images(images, epoch, iteration, directory=save_image_folder):
    os.makedirs(directory, exist_ok=True)
    for i, img in enumerate(images):
        img = img.permute(1, 2, 0).cpu().detach().numpy()
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1]
        plt.imsave(os.path.join(directory, f'epoch_{epoch}_iter_{iteration}_img_{i}.png'), img)
		

# Saving Generator Model
def save_generator_model(generator, epoch, folder=save_model_folder):
    os.makedirs(folder, exist_ok=True)
    torch.save(generator.state_dict(), os.path.join(folder, f'generator_epoch_{epoch}.pth'))

best_accuracy = 0.0
best_epoch = 0
# Instantiate models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Define optimizers
optimizerG = torch.optim.Adam(generator.parameters(), lr=learning_rate)
optimizerD = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

# Loss function
criterion = nn.BCELoss()

# Training Loop
# Initial augmentation probability
p = 0.1
# Control variables for adjustment
adjustment_interval = 100  # Number of batches after which to adjust p
performance_target = 0.8  # Target performance for the discriminator
# traning loop
for epoch in range(epochs):
    for i, images in enumerate(dataloader):
        images = images.to(device)
        real_labels = torch.ones(images.size(0), device=device)
        fake_labels = torch.zeros(images.size(0), device=device)

        # --- Train Discriminator ---
        optimizerD.zero_grad()

        # Train with real images
        real_images = images
        real_loss = criterion(discriminator(real_images), real_labels)

        # Train with fake images
        z = generate_latent_points(images.size(0), latent_dim).to(device)
        fake_images = generator(z)
        fake_images_augmented = apply_augmentations(fake_images, p)
        fake_loss = criterion(discriminator(fake_images_augmented.detach()), fake_labels)

        # Update discriminator
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizerD.step()

        # --- Train Generator ---
        optimizerG.zero_grad()

        # Generate fake images trying to fool the discriminator
        z = generate_latent_points(images.size(0), latent_dim).to(device)
        fake_images = generator(z)
        fake_images_augmented = apply_augmentations(fake_images, p)
        g_loss = criterion(discriminator(fake_images_augmented), real_labels)

        # Update generator
        g_loss.backward()
        optimizerG.step()

        # Adjust augmentation probability p based on the discriminator's performance
        if (i + 1) % adjustment_interval == 0:
            accuracy = evaluate_performance(discriminator, dataloader, generator, device)
            if accuracy > performance_target:
                p = min(p + 0.05, 1.0)
            else:
                p = max(p - 0.05, 0.0)

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}, p: {p}')

    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            test_images = generator(generate_latent_points(1, latent_dim))  # Generate one image
        save_generated_images(test_images, epoch, 0)

print("Training complete")
