import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import torch.nn.functional as F

# Parameters
batch_size = 20
image_size = 128
learning_rate = 0.0002
epochs = 50
timesteps = 1000  # Number of diffusion steps
beta_schedule = np.linspace(0.001, 0.2, timesteps)  # Variance schedule
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
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)  

# Update save paths based on selected dataset
save_model_folder = f'home/shassa01/THESIS/Code/DS1savedModels/{selected_dataset}'
save_image_folder = f'/home/shassa01/THESIS/Code/DGS1-images/{selected_dataset}'

# UNet-like architecture for the diffusion model
class DiffusionModel(nn.Module):
    def __init__(self, time_embedding_size=64):
        super(DiffusionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )
        self.time_embedding = nn.Embedding(timesteps, time_embedding_size)
        self.time_fc = nn.Linear(time_embedding_size, 64 * 8 * 8)  # Output shape: [batch_size, 64, 8, 8]

        # Adjusted upsampling layers to reach the desired size incrementally
        self.fc_to_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # Output shape: [batch_size, 64, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output shape: [batch_size, 32, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # Output shape: [batch_size, 16, 64, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)    # Output shape: [batch_size, 3, 128, 128]
        )

    def forward(self, x, t):
        t = t.long()
        t_embed = self.time_embedding(t)
        t_processed = self.time_fc(t_embed)
        t_processed = t_processed.view(-1, 64, 8, 8)  # Reshape to match the initial feature map size

        t_processed = self.fc_to_conv(t_processed)  # Upsample to match input image dimensions

        if t_processed.size() != x.size():
            raise ValueError(f"Size mismatch: {t_processed.size()} vs {x.size()}")

        x_with_time = x + t_processed
        return self.model(x_with_time)


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
    os.makedirs(folder, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(folder, f'model_epoch_{epoch}.pth'))

# Training Loop
model = DiffusionModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) 

for epoch in range(epochs):
    for images in dataloader:
        current_batch_size = images.size(0)  
        images = images.to(device)
        for t in range(timesteps - 1, -1, -1):
            optimizer.zero_grad()
            # Calculate the current noise level
            current_noise_level = noise_level(t, beta_schedule)
            # Add noise to images
            noisy_images = add_noise(images, current_noise_level)
            # Predict the denoised image at this timestep
            t_tensor = torch.tensor([t] * current_batch_size, device=device)  # Adjust to current batch size
            predicted_images = model(noisy_images, t_tensor)
            # Calculate loss (e.g., MSE between predicted denoised images and original images)
            loss = nn.MSELoss()(predicted_images, images)
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')
        scheduler.step()
        generate_and_save_images(model, epoch)
        save_model(model, epoch)  

print("Training complete")
