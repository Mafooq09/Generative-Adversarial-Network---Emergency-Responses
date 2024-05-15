import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Parameters
batch_size = 5
image_size = 128
learning_rate = 0.0002
epochs = 700
timesteps = 1000  # Number of diffusion steps
beta_schedule = np.linspace(0.0001, 0.02, timesteps)  # Variance schedule
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

# Dataset loading
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
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Update save paths based on selected dataset
save_model_folder = f'home/shassa01/THESIS/Code/DsavedModels/{selected_dataset}'
save_image_folder = f'/home/shassa01/THESIS/Code/DG-images/{selected_dataset}'

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class DiffusionModel(nn.Module):
    def __init__(self, image_channels=3, features=[64, 128, 256, 512], time_embedding_size=64):
        super().__init__()
        self.initial_conv = ConvBlock(image_channels, features[0], kernel_size=7, padding=3)

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.res_blocks = nn.ModuleList()

        in_channels = features[0]
        for feature in features[1:]:
            self.down_blocks.append(ConvBlock(in_channels, feature, kernel_size=3, stride=2, padding=1))
            self.res_blocks.append(ResidualBlock(feature))
            in_channels = feature

        for feature in reversed(features[1:]):
            self.up_blocks.append(
                ConvBlock(in_channels, feature, down=False, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
            self.res_blocks.append(ResidualBlock(feature))
            in_channels = feature

        self.final_conv = nn.Conv2d(in_channels, image_channels, kernel_size=7, padding=3)

        self.time_embedding = nn.Embedding(timesteps, time_embedding_size)
        self.time_fc = nn.Linear(time_embedding_size, features[0] * 2)

    def forward(self, x, t):
        t_embed = self.time_embedding(t)
        t_processed = self.time_fc(t_embed)
        t_processed = t_processed.view(t_processed.shape[0], 2, -1, 1, 1)

        x = self.initial_conv(x)
        x = x + t_processed[:, 0]

        skip_connections = []

        for down, res in zip(self.down_blocks, self.res_blocks[:len(self.down_blocks)]):
            x = down(x)
            x = res(x)
            skip_connections.append(x)

        skip_connections = skip_connections[::-1]

        for up, res, skip in zip(self.up_blocks, self.res_blocks[len(self.down_blocks):], skip_connections):
            x = up(x)
            x = res(x)
            x = x + skip

        x = self.final_conv(x)
        x = x + t_processed[:, 1]

        return torch.tanh(x)
def add_noise(images, noise_level):
    return images + noise_level * torch.randn_like(images)

def noise_level(t, beta_schedule):
    return torch.sqrt(torch.tensor(beta_schedule[t], device=device))

def generate_and_save_images(model, epoch, num_images=5, save_dir=save_image_folder):
    # Ensure the model is in evaluation mode to deactivate dropout layers, etc.
    model.eval()
    
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Generate images
    for i in range(num_images):
        # Start with pure noise
        noise = torch.randn(1, 3, image_size, image_size, device=device)
        
        # Initialize the image as noise
        image = noise.clone()
        
        # Reverse the diffusion process
        with torch.no_grad():
            for t in reversed(range(timesteps)):
                t_tensor = torch.tensor([t], device=device)
                image = model(image, t_tensor)
        
        # Post-process the image
        image = image.squeeze(0)  # Remove batch dimension
        image = (image + 1) / 2  # Rescale values from [-1, 1] to [0, 1]
        image = image.clamp(0, 1)  # Ensure values are in [0, 1]
        image = transforms.ToPILImage()(image.cpu())

        # Create a unique filename for each saved image
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(save_dir, f'epoch_{epoch}_image_{i}_{current_time}.png')
        
        # Save the image
        image.save(filename)

    # Set the model back to training mode
    model.train()

def save_model(model, epoch, folder=save_model_folder):
    if epoch % 2 == 0:  # Adjust the frequency of model saving as needed
        os.makedirs(folder, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(folder, f'model_epoch_{epoch}.pth'))

# Training Loop
model = DiffusionModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for images in dataloader:
        images = images.to(device)
        for t in range(timesteps - 1, -1, -1):
            optimizer.zero_grad()
            # Calculate the current noise level
            current_noise_level = noise_level(t, beta_schedule)
            # Add noise to images
            noisy_images = add_noise(images, current_noise_level)
            # Predict the denoised image at this timestep
            t_tensor = torch.tensor([t] * batch_size, device=device)
            predicted_images = model(noisy_images, t_tensor)
            # Calculate loss (e.g., MSE between predicted denoised images and original images)
            loss = nn.MSELoss()(predicted_images, images)
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')
            generate_and_save_images(model, epoch)
            save_model(model, epoch)  

print("Training complete")
