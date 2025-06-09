import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import cv2
import os

import torch
import torch.nn as nn

class PatchAutoencoderCNN(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, stride):
        super(PatchAutoencoderCNN, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.stride = stride
        
        self.num_patches = ((img_size - patch_size) // stride + 1) ** 2
        self.patch_area = patch_size * patch_size

        # Unfold and fold for patch extraction and reconstruction
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=stride)
        self.fold = nn.Fold(
            output_size=(img_size, img_size),
            kernel_size=patch_size,
            stride=stride
        )

        # CNN encoder for R, G, B (input: (B*num_patches, 1, patch_size, patch_size))
        def cnn_encoder():
            return nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 1, kernel_size=3, padding=1),
                nn.ReLU()
            )
        self.cnn_enc_r = cnn_encoder()
        self.cnn_enc_g = cnn_encoder()
        self.cnn_enc_b = cnn_encoder()

        # Linear encoders
        self.encoder_r = nn.Linear(self.patch_area, embed_dim)
        self.encoder_g = nn.Linear(self.patch_area, embed_dim)
        self.encoder_b = nn.Linear(self.patch_area, embed_dim)

        # Linear decoders
        self.decoder_r = nn.Linear(embed_dim, self.patch_area)
        self.decoder_g = nn.Linear(embed_dim, self.patch_area)
        self.decoder_b = nn.Linear(embed_dim, self.patch_area)

        # CNN decoder after linear (same structure)
        def cnn_decoder():
            return nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 1, kernel_size=3, padding=1),
                nn.ReLU()
            )
        self.cnn_dec_r = cnn_decoder()
        self.cnn_dec_g = cnn_decoder()
        self.cnn_dec_b = cnn_decoder()

    def apply_cnn(self, patches, cnn):
        # patches: (B, num_patches, patch_area)
        B, N, D = patches.shape
        patches = patches.reshape(B * N, 1, self.patch_size, self.patch_size)
        patches = cnn(patches)  # (B*N, 1, patch_size, patch_size)
        return patches.reshape(B, N, -1)

    def forward(self, x):
        B = x.size(0)

        # Extract patches → (B, 3*patch_area, num_patches)
        patches = self.unfold(x)
        patches = patches.permute(0, 2, 1)  # (B, num_patches, 3*patch_area)

        # Split RGB
        r, g, b = torch.chunk(patches, 3, dim=2)  # (B, N, patch_area)

        # CNN + Linear encoder
        r = self.apply_cnn(r, self.cnn_enc_r)
        g = self.apply_cnn(g, self.cnn_enc_g)
        b = self.apply_cnn(b, self.cnn_enc_b)
        emb_r = self.encoder_r(r)
        emb_g = self.encoder_g(g)
        emb_b = self.encoder_b(b)

        # Linear decoder
        rec_r = self.decoder_r(emb_r)
        rec_g = self.decoder_g(emb_g)
        rec_b = self.decoder_b(emb_b)

        # CNN decoder
        rec_r = self.apply_cnn(rec_r, self.cnn_dec_r)
        rec_g = self.apply_cnn(rec_g, self.cnn_dec_g)
        rec_b = self.apply_cnn(rec_b, self.cnn_dec_b)

        # Concatenate and fold
        rec = torch.cat([rec_r, rec_g, rec_b], dim=2).permute(0, 2, 1)
        x_recon = self.fold(rec)

        # Normalize overlaps
        ones = torch.ones((B, 3, self.img_size, self.img_size), device=x.device)
        norm = self.fold(self.unfold(ones))
        return x_recon / norm

    def encode(self, x):
        patches = self.unfold(x).permute(0, 2, 1)
        r, g, b = torch.chunk(patches, 3, dim=2)
        r = self.apply_cnn(r, self.cnn_enc_r)
        g = self.apply_cnn(g, self.cnn_enc_g)
        b = self.apply_cnn(b, self.cnn_enc_b)
        emb_r = self.encoder_r(r)
        emb_g = self.encoder_g(g)
        emb_b = self.encoder_b(b)
        return torch.cat([emb_r, emb_g, emb_b], dim=2)

    def decode(self, emb):
        emb_r, emb_g, emb_b = torch.chunk(emb, 3, dim=2)
        rec_r = self.decoder_r(emb_r)
        rec_g = self.decoder_g(emb_g)
        rec_b = self.decoder_b(emb_b)
        rec_r = self.apply_cnn(rec_r, self.cnn_dec_r)
        rec_g = self.apply_cnn(rec_g, self.cnn_dec_g)
        rec_b = self.apply_cnn(rec_b, self.cnn_dec_b)
        rec = torch.cat([rec_r, rec_g, rec_b], dim=2).permute(0, 2, 1)
        x_recon = self.fold(rec)
        B = emb.size(0)
        norm = self.fold(self.unfold(torch.ones((B, 3, self.img_size, self.img_size), device=emb.device)))
        return x_recon / norm

class PatchAutoencoder(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, stride):
        super(PatchAutoencoder, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.stride = stride
        
        self.num_patches = ((img_size - patch_size) // stride + 1) ** 2
        self.patch_area = patch_size * patch_size  # area of one channel patch

        # Unfold and fold for patch extraction and reconstruction
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=stride)
        self.fold = nn.Fold(
            output_size=(img_size, img_size),
            kernel_size=patch_size,
            stride=stride
        )

        # Separate encoder/decoder for each channel
        self.encoder_r = nn.Linear(self.patch_area, embed_dim)
        self.encoder_g = nn.Linear(self.patch_area, embed_dim)
        self.encoder_b = nn.Linear(self.patch_area, embed_dim)

        self.decoder_r = nn.Linear(embed_dim, self.patch_area)
        self.decoder_g = nn.Linear(embed_dim, self.patch_area)
        self.decoder_b = nn.Linear(embed_dim, self.patch_area)

    def forward(self, x):
        B = x.size(0)

        # Extract patches → (B, 3*patch_area, num_patches)
        patches = self.unfold(x)
        patches = patches.permute(0, 2, 1)  # (B, num_patches, 3*patch_area)

        # Split into R, G, B channel patches
        r, g, b = torch.chunk(patches, chunks=3, dim=2)  # each (B, num_patches, patch_area)

        # Encode per channel
        emb_r = self.encoder_r(r)
        emb_g = self.encoder_g(g)
        emb_b = self.encoder_b(b)

        # Decode per channel
        rec_r = self.decoder_r(emb_r)
        rec_g = self.decoder_g(emb_g)
        rec_b = self.decoder_b(emb_b)

        # Concatenate reconstructed patches → (B, num_patches, 3*patch_area)
        rec_patches = torch.cat([rec_r, rec_g, rec_b], dim=2)
        rec_patches = rec_patches.permute(0, 2, 1)  # (B, 3*patch_area, num_patches)

        # Fold back to image
        x_recon = self.fold(rec_patches)

        # Normalize overlaps
        ones = torch.ones((B, 3, self.img_size, self.img_size), device=x.device)
        norm = self.fold(self.unfold(ones))
        return x_recon / norm

    def encode(self, x):
        patches = self.unfold(x).permute(0, 2, 1)
        r, g, b = torch.chunk(patches, 3, dim=2)
        emb_r = self.encoder_r(r)
        emb_g = self.encoder_g(g)
        emb_b = self.encoder_b(b)
        return torch.cat([emb_r, emb_g, emb_b], dim=2)  # (B, num_patches, 3*embed_dim)

    def decode(self, emb):
        emb_r, emb_g, emb_b = torch.chunk(emb, 3, dim=2)
        rec_r = self.decoder_r(emb_r)
        rec_g = self.decoder_g(emb_g)
        rec_b = self.decoder_b(emb_b)
        rec_patches = torch.cat([rec_r, rec_g, rec_b], dim=2).permute(0, 2, 1)
        x_recon = self.fold(rec_patches)
        B = emb.size(0)
        dummy = torch.ones((B, 3, self.img_size, self.img_size), device=emb.device)
        norm = self.fold(self.unfold(dummy))
        return x_recon / norm
# -----------------------------------------------------------------------------
# Training Function
# -----------------------------------------------------------------------------
def train_model(model, loader, criterion, optimizer, device, num_epochs, checkpoint_dir):
    model.train()
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(loader):
            images = batch.to(device)  # (B, 3, img_size, img_size)
            
            # Forward pass
            recon = model(images)
            loss = criterion(recon, images)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if (step + 1) % 10 == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}] Step [{step+1}/{len(loader)}]  Loss: {loss.item():.6f}")
        
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}\n")
        
        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"patch_autoencoder_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

# -----------------------------------------------------------------------------
# Inference Function (with embedding printout)
# -----------------------------------------------------------------------------
def inference_model(model, loader, device, num_to_display=8):
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        images = batch.to(device)
        
        # 1) Get reconstructions
        recon_images = model(images)
        
        # 2) Print embeddings for the first sample in the batch
        emb = model.encode(images)  # (B, num_patches, embed_dim)
        print("Embeddings for first image in the batch (shape):", emb[0].shape)
        print("Embedding vector (first patch) for first image:\n", emb[0, 0, :10], "...")  # print first 10 dims as example
        
        # Move to CPU & convert to NumPy
        images_np = images.cpu().permute(0, 2, 3, 1).numpy()      # (B, img_size, img_size, 3)
        recon_np  = recon_images.cpu().permute(0, 2, 3, 1).numpy() # (B, img_size, img_size, 3)
        
        # Display first num_to_display examples side-by-side using OpenCV
        num_display = min(num_to_display, images_np.shape[0])
        for idx in range(num_display):
            orig = (images_np[idx] * 255).astype(np.uint8)
            recon = np.clip(recon_np[idx] * 255, 0, 255).astype(np.uint8)
            combined = np.concatenate([orig, recon], axis=1)
            cv2.imshow("", combined)
            key = cv2.waitKey(1000)
            if key == 27: 
                break
        
        cv2.destroyAllWindows()

# -----------------------------------------------------------------------------
# Helper Functions to Encode & Decode with Saved Checkpoint
# -----------------------------------------------------------------------------
def encode_image(image_tensor, checkpoint_path, img_size, patch_size, embed_dim, stride, device):
    """
    Given a single image tensor of shape (3, img_size, img_size),
    loads the model from checkpoint_path and returns its embedding.
    """
    model = PatchAutoencoder(img_size, patch_size, embed_dim, stride).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Ensure shape is (1, 3, img_size, img_size)
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        emb = model.encode(image_tensor)  # (1, num_patches, embed_dim)
    return emb.squeeze(0).cpu()  # return (num_patches, embed_dim) on CPU

def decode_embedding(embedding_tensor, checkpoint_path, img_size, patch_size, embed_dim, stride, device):
    """
    Given an embedding tensor of shape (num_patches, embed_dim),
    loads the model from checkpoint_path and returns the reconstructed image tensor.
    """
    model = PatchAutoencoder(img_size, patch_size, embed_dim, stride).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Ensure shape is (1, num_patches, embed_dim)
    if embedding_tensor.dim() == 2:
        embedding_tensor = embedding_tensor.unsqueeze(0)
    embedding_tensor = embedding_tensor.to(device)
    
    with torch.no_grad():
        recon = model.decode(embedding_tensor)  # (1, 3, img_size, img_size)
    return recon.squeeze(0).cpu()  # return (3, img_size, img_size) on CPU

# -----------------------------------------------------------------------------
# Main script: dataset loading, training loop, and inference/display
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # ------------------------ Hyperparameters ------------------------
    img_size = 224           # Assuming images are 224×224
    patch_size = 16          # Each patch is 16×16
    embed_dim = 128          # Embedding dimension for each patch
    overlap_ratio = 0.5     # Overlap ratio between patches
    stride = 13#max(1, int(patch_size * (1 - overlap_ratio)))
    
    num_epochs = 1
    batch_size = 128
    learning_rate = 1e-4
    checkpoint_dir = 'checkpoints'
    
    # ------------------------ Imports for Dataset ------------------------
    # Assume Image_Dataset is defined elsewhere; it should yield tensors of shape (3, 224, 224)
    from torchvision import transforms
    from dataloaders import Image_Dataset  # <-- replace with actual module
    
    # ------------------------ DataLoader ------------------------
    transform = transforms.ToTensor()
    ds = Image_Dataset('out_dataset_bottle', transform=transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2
    )
    
    # Print one batch shape as a sanity check
    sample_batch = next(iter(loader))
    print(f"Sample batch tensor shape: {sample_batch.shape}")  # Expect (B, 3, 224, 224)
    
    # ------------------------ Device Setup ------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ------------------------ Model, Loss, Optimizer ------------------------
    model = PatchAutoencoder(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        stride=stride
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # ------------------------ Train ------------------------
    # train_model(
    #     model=model,
    #     loader=loader,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     device=device,
    #     num_epochs=num_epochs,
    #     checkpoint_dir=checkpoint_dir
    # )
    
    # ------------------------ Inference & Display ------------------------
    # Load the last checkpoint
    last_checkpoint = os.path.join(checkpoint_dir, f"patch_autoencoder_epoch{num_epochs}.pth")
    model.load_state_dict(torch.load(last_checkpoint, map_location=device))
    inference_model(model, loader, device, num_to_display=128)
    
    # ------------------------ Example Usage of encode_image & decode_embedding ------------------------
    # (Uncomment and adjust below as needed)
    # from PIL import Image
    # img = Image.open('path_to_single_image.jpg').convert('RGB').resize((img_size, img_size))
    # img_tensor = transforms.ToTensor()(img)  # shape: (3, img_size, img_size)
    # 
    # # Encode a single image to embeddings
    # embeddings = encode_image(
    #     image_tensor=img_tensor,
    #     checkpoint_path=last_checkpoint,
    #     img_size=img_size,
    #     patch_size=patch_size,
    #     embed_dim=embed_dim,
    #     stride=stride,
    #     device=device
    # )
    # print("Embeddings shape:", embeddings.shape)  # (num_patches, embed_dim)
    # 
    # # Decode embeddings back to image
    # reconstructed_img_tensor = decode_embedding(
    #     embedding_tensor=embeddings,
    #     checkpoint_path=last_checkpoint,
    #     img_size=img_size,
    #     patch_size=patch_size,
    #     embed_dim=embed_dim,
    #     stride=stride,
    #     device=device
    # )
    # 
    # # Convert to NumPy and display via OpenCV
    # recon_np = reconstructed_img_tensor.permute(1, 2, 0).numpy()  # (img_size, img_size, 3)
    # recon_np_uint8 = np.clip(recon_np * 255, 0, 255).astype(np.uint8)
    # cv2.imshow("Reconstructed from Embedding", cv2.cvtColor(recon_np_uint8, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
