# test_pipeline.py

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# -------------------------
# Fix import paths
# -------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from model.vae_2 import VAE
from src.config import Config


# -------------------------
# Hardcoded Test Images
# -------------------------
TEST_IMAGES = [
    ("E:\\SecurePix_Dataset\\CLEAN_IMGAES\\6e2dg.png", 0),
    ("E:\\SecurePix_Dataset\\CLEAN_IMGAES\\3p67n.png", 0),
    ("E:\\SecurePix_Dataset\\CLEAN_IMGAES\\25w53.png", 0),
    ("E:\\SecurePix_Dataset\\CLEAN_IMGAES\\gcx6f.png", 0),
    ("E:\\SecurePix_Dataset\\CLEAN_IMGAES\\y5dpp.png", 0),
    ("E:\\SecurePix_Dataset\\CLEAN_IMGAES\\w2n7e.png", 0),
    ("E:\\SecurePix_Dataset\\STEG_LSB1_IMAGES\\gnc3n.png", 1),
    ("E:\\SecurePix_Dataset\\STEG_LSB1_IMAGES\\d66cn.png", 1),
    ("E:\\SecurePix_Dataset\\STEG_LSB3_IMAGES\\efe62.png", 1),
    ("E:\\SecurePix_Dataset\\STEG_LSB3_IMAGES\\ddcne.png", 1)
]

# -------------------------
# Utility: Convert image to SRM stack
# -------------------------
def load_image(img_path):
    img = Image.open(img_path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # 1 channel
    arr = np.repeat(arr, Config.IMG_CHANNELS, axis=0)  # replicate channels
    return torch.tensor(arr, dtype=torch.float32)

# -------------------------
# Reconstruction Visualization
# -------------------------
def save_reconstruction(x, x_hat, file_name):
    x = x.numpy()
    x_hat = x_hat.numpy()
    x_vis = x.mean(axis=0)
    xhat_vis = x_hat.mean(axis=0)
    diff = np.abs(x_vis - xhat_vis)

    plt.figure(figsize=(10,3))
    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(x_vis, cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title("Reconstruction")
    plt.imshow(xhat_vis, cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title("Error Map")
    plt.imshow(diff, cmap='hot')
    plt.axis('off')

    plt.savefig(file_name)
    plt.close()
    return file_name

# -------------------------
# Load Model
# -------------------------
model = VAE(input_channels=Config.IMG_CHANNELS, latent_dim=Config.LATENT_DIM)
model.load_state_dict(torch.load(
    os.path.join(Config.CHECKPOINTS_DIR, Config.CHECKPOINT_NAME),
    map_location=Config.DEVICE
))
model = model.to(Config.DEVICE)
model.eval()

# -------------------------
# Compute Threshold
# -------------------------
# For demo, using mean + 2*std of clean images in this batch
clean_errors = []
for img_path, label in TEST_IMAGES:
    if label == 0:
        x = load_image(img_path).unsqueeze(0).to(Config.DEVICE)
        with torch.no_grad():
            x_hat, _, _ = model(x)
            err = ((x - x_hat) ** 2).mean().item()
            clean_errors.append(err)
threshold = np.mean(clean_errors) + 2 * np.std(clean_errors)

# -------------------------
# PDF Report Setup
# -------------------------
pdf_path = os.path.join(Config.RESULTS_DIR, "test_1.pdf")
os.makedirs(Config.RESULTS_DIR, exist_ok=True)
pdf = canvas.Canvas(pdf_path, pagesize=letter)

# -------------------------
# Inference Loop
# -------------------------
for idx, (img_path, label) in enumerate(TEST_IMAGES):

    x = load_image(img_path).unsqueeze(0).to(Config.DEVICE)
    with torch.no_grad():
        x_hat, _, _ = model(x)
        error = ((x - x_hat) ** 2).mean().item()
        pred = 1 if error > threshold else 0

    # Save reconstruction image
    recon_img_path = os.path.join(Config.RESULTS_DIR, f"recon_{idx}.png")
    save_reconstruction(x.cpu()[0], x_hat.cpu()[0], recon_img_path)

    # -------------------------
    # Add to PDF
    # -------------------------
    pdf.drawString(50, 750, f"Image: {os.path.basename(img_path)}")
    pdf.drawString(50, 730, f"Ground Truth: {'STEGO' if label==1 else 'CLEAN'}")
    pdf.drawString(50, 710, f"Prediction: {'STEGO' if pred==1 else 'CLEAN'}")
    pdf.drawString(50, 690, f"Reconstruction Error: {error:.6f}")
    pdf.drawImage(recon_img_path, 50, 300, width=500, height=200)
    pdf.showPage()

pdf.save()
print(f"Test report saved at: {pdf_path}")