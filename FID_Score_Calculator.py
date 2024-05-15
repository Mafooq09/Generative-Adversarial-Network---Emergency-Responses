import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3, Inception_V3_Weights


class InceptionFeatureExtractor(torch.nn.Module):
    def __init__(self, inception_model):
        super(InceptionFeatureExtractor, self).__init__()
        self.features = torch.nn.Sequential(*list(inception_model.children())[:-1])

    def forward(self, x):
        print("Initial shape:", x.shape)
        x = self.features(x)
        print("After features shape:", x.shape)
        if x.size(2) != 1 or x.size(3) != 1:
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        print("After pooling shape:", x.shape)
        x = torch.flatten(x, start_dim=1)
        print("Final flattened shape:", x.shape)
        return x


# Function to load images
def load_images(path):
    images = []
    filenames = os.listdir(path)
    for filename in filenames:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(os.path.join(path, filename)).convert('RGB')
            images.append(image)
    return images


# Function to preprocess images for Inception model
def preprocess_images(images, image_size=299):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor_images = torch.stack([transform(image) for image in images])
    return tensor_images


# Function to calculate activations using Inception model
def get_activations(images, model, batch_size=10, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    print("Input batch shape:", images.shape)
    model.eval()
    num_images = images.size(0)
    n_batches = num_images // batch_size + (1 if num_images % batch_size != 0 else 0)
    activations = []
    for i in range(n_batches):
        batch = images[i * batch_size: (i + 1) * batch_size].to(device)
        with torch.no_grad():
            pred = model(batch)
            print("Batch output shape:", pred.shape)  # Debug output shape
            activations.append(pred.detach().cpu().numpy())
    activations = np.concatenate(activations, axis=0)
    return activations


# Function to calculate the FID score
def calculate_fid(real_activations, fake_activations):
    print(f"Real activations shape: {real_activations.shape}")  # Debug
    print(f"Fake activations shape: {fake_activations.shape}")  # Debug
    mu1, sigma1 = real_activations.mean(axis=0), np.cov(real_activations, rowvar=False)
    mu2, sigma2 = fake_activations.mean(axis=0), np.cov(fake_activations, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


# Load Inception model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).to(device)


def main(real_path, fake_path):
    real_images = load_images(real_path)
    fake_images = load_images(fake_path)

    preprocessed_real_images = preprocess_images(real_images)
    preprocessed_fake_images = preprocess_images(fake_images)

    # Initialize and use the feature extractor instead of the base Inception model
    inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).to(device)
    feature_extractor = InceptionFeatureExtractor(inception_model).to(device)

    real_activations = get_activations(preprocessed_real_images, feature_extractor)
    fake_activations = get_activations(preprocessed_fake_images, feature_extractor)

    fid_score = calculate_fid(real_activations, fake_activations)
    print('Scenario 3: Original + Generated Dataset against Generated Images')
    print(f'FID score: {fid_score}')


if __name__ == "__main__":
    real_image_path = '/home/shassa01/THESIS/AIDER/fire'
    generated_image_path = '/home/shassa01/THESIS/Code/G-images/fire'
    main(real_image_path, generated_image_path)
