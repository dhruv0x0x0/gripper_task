import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloaders import Image_Dataset
# Define the transformation
transform = transforms.Compose([
  #  transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),          # Convert images to PyTorch tensors
])

# Initialize your dataset and dataloader
ds2 = Image_Dataset('out_dataset_bottle', transform=transform)
loader = DataLoader(
    ds2,
    batch_size=10,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=2
)
from diffusers import AutoencoderKL

class StableDiffusionVAE(torch.nn.Module):
    def __init__(self, model_path="stabilityai/sd-vae-ft-mse"):
        super().__init__()
        self.scaling_factor = 0.18215
        self.vae = AutoencoderKL.from_pretrained(model_path).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).eval()

    def encode(self, x):
        with torch.no_grad():
            latent = self.vae.encode(x).latent_dist.sample()
            return latent * self.scaling_factor

    def decode(self, z):
        z = z / self.scaling_factor
        with torch.no_grad():
            return self.vae.decode(z).sample

import matplotlib.pyplot as plt
import numpy as np

# Initialize the VAE
vae = StableDiffusionVAE()

# Process a batch of images
for batch in loader:
    # Ensure the batch is on the correct device
    batch = batch.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Encode and decode the images
    latents = vae.encode(batch)
    print(latents.shape)
    print(latents[0])
    print(latents[1])
    reconstructions = vae.decode(latents)

    # Move tensors to CPU and convert to NumPy arrays
    originals = batch.cpu().permute(0, 2, 3, 1).numpy()
    reconstructions = reconstructions.cpu().permute(0, 2, 3, 1).numpy()

    # Display the first 5 original and reconstructed images
    for i in range(5):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        orig_rgb = originals[i][..., ::-1]           # BGR to RGB
        recon_rgb = reconstructions[i][..., ::-1]  
        axes[0].imshow(np.clip(orig_rgb, 0, 1))
        axes[0].set_title('Original')
        axes[0].axis('off')
        axes[1].imshow(np.clip(recon_rgb, 0, 1))
        axes[1].set_title('Reconstructed')
        axes[1].axis('off')
        plt.show()
    break  # Remove this break to process more batches
