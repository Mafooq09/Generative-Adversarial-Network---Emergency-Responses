import torch
from torchvision import transforms
from torch import nn
import matplotlib.pyplot as plt
import os
from PIL import Image

# Parameters
image_size = 128
latent_dim = 100
device = torch.device("cpu")  # Change to "cuda:0" if GPU is available and desired

# Define the Generator architecture (this must match the original model's architecture)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Load the trained model
model_path = 'E:/THESIS/Code/fire/generator_epoch_119.pth'
generator = Generator().to(device)
generator.load_state_dict(torch.load(model_path, map_location=device))
generator.eval()  # Set the model to evaluation mode

# Generate images
def generate_images(num_images):
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim, 1, 1, device=device)
        generated_images = generator(z)
        generated_images = generated_images * 0.5 + 0.5  # Transform images to [0,1] from [-1,1]
        return generated_images

def save_generated_images(images, directory='E:/THESIS/Code/generated_images'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i, image in enumerate(images):
        plt.figure(figsize=(2.5, 2.5))
        plt.axis('off')
        plt.imshow(image.permute(1, 2, 0).cpu().numpy())  # Convert from CHW to HWC format
        plt.savefig(f"{directory}/generated_image_{i+1}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

# Generate and save images
num_images = 200  # Specify the number of images to generate
generated_images = generate_images(num_images)
save_generated_images(generated_images)

print(f"Generated and saved {num_images} images to the specified directory.")
