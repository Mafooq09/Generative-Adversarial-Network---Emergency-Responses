import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


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
save_model_folder = f'home/shassa01/THESIS/Code/savedModels/{selected_dataset}'
save_image_folder = f'/home/shassa01/THESIS/Code/G-images/{selected_dataset}'

# UNet-like architecture for the diffusion model
class DiffusionModel(nn.Module):
    def __init__(self, time_embedding_size=64):
        super(DiffusionModel, self).__init__()
        # Model similar to what you previously defined
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )
        
        # Additional layers for processing the time step embedding
        self.time_embedding = nn.Embedding(timesteps, time_embedding_size)
        self.time_fc = nn.Linear(time_embedding_size, 3 * 64 * 64)  # Adjust the size to match your image dimensions

    def forward(self, x, t):
        # x: input image, t: time step
        # Embed the time step
        t_embed = self.time_embedding(t)
        
        # Process the time embedding to match the spatial dimensions of the image
        t_processed = self.time_fc(t_embed)
        t_processed = t_processed.view(-1, 3, 64, 64)  # Reshape to match image dimensions (batch_size, channels, height, width)
        
        # Combine the time information with the input image. Here, we simply add them, but other methods (e.g., concatenation) are also possible.
        x_with_time = x + t_processed
        
        # Pass the combined input through the model
        return self.model(x_with_time)

# Function to add noise to images
def add_noise(images, noise_level):
    return images + noise_level * torch.randn_like(images)

# Function to calculate noise level for a given step
def noise_level(t, beta_schedule):
    return torch.sqrt(torch.tensor(beta_schedule[t], device=device))
    
def generate_and_save_images(model, epoch, num_images=5, save_dir='generated_images'):
    model.eval()  # Set the model to evaluation mode
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    for i in range(num_images):
        # Start with pure noise
        image = torch.randn(1, 3, image_size, image_size, device=device)
        with torch.no_grad():
            for t in reversed(range(timesteps)):
                # Calculate the noise level for this timestep
                current_noise_level = noise_level(t, beta_schedule)
                # Apply the model to reduce the noise level
                image = model(image, t) - current_noise_level * torch.randn_like(image)

        # Convert the tensor to a PIL image and save
        image = image.squeeze(0)  # Remove the batch dimension
        image = (image + 1) / 2  # Rescale image values from [-1, 1] to [0, 1]
        image = image.clamp(0, 1)  # Clamp values to [0, 1] to ensure valid image
        image = transforms.ToPILImage()(image.cpu())
        image.save(os.path.join(save_dir, f'epoch_{epoch}_image_{i}.png'))

    model.train()

# Training Loop
model = DiffusionModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for images in dataloader:
        images = images.to(device)
        for t in range(timesteps - 1, -1, -1):  # Reverse loop over timesteps
            optimizer.zero_grad()
            # Calculate the current noise level
            current_noise_level = noise_level(t, beta_schedule)
            # Add noise to images
            noisy_images = add_noise(images, current_noise_level)
            # Predict the noise (or denoised image) at this timestep
            predicted_noise = model(noisy_images, t)
            # Calculate loss (e.g., MSE between predicted noise and actual noise)
            loss = nn.MSELoss()(predicted_noise, noisy_images - images)
            loss.backward()
            optimizer.step()
    if epoch % 10 == 0:  # Generate and save images every 10 epochs
        generate_and_save_images(model, epoch)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

print("Training complete")
