# executor.py
import os
import sys
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# -------------------------
# Fix import paths
# -------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from model.vae_2 import VAE
from src.config import Config

# -------------------------
# Setup logging
# -------------------------
os.makedirs(Config.EXECUTION_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logging.info("Executor started.")

# -------------------------
# Load VAE model
# -------------------------
checkpoint_path = os.path.join(Config.CHECKPOINTS_DIR, Config.CHECKPOINT_NAME)
logging.info(f"Loading VAE model from checkpoint: {checkpoint_path}")
model = VAE(input_channels=Config.IMG_CHANNELS, latent_dim=Config.LATENT_DIM)
model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
model = model.to(Config.DEVICE)
model.eval()

#print space to evade the warning msg
print("\n" * 2)

logging.info("Model loaded and set to evaluation mode.")

# -------------------------
# Utility: Load & preprocess image
# -------------------------
def load_image(img_path):
    logging.info(f"Loading image: {img_path}")
    img = Image.open(img_path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # 1 channel
    arr = np.repeat(arr, Config.IMG_CHANNELS, axis=0)
    tensor = torch.tensor(arr, dtype=torch.float32)
    logging.info(f"Image shape after preprocessing: {tensor.shape}")
    return tensor

# -------------------------
# Compute reconstruction + error map
# -------------------------
def run_inference(img_path, threshold):
    x = load_image(img_path).unsqueeze(0).to(Config.DEVICE)
    with torch.no_grad():
        x_hat, _, _ = model(x)
        error_map = ((x - x_hat) ** 2).mean(dim=1, keepdim=True)
        recon_error = error_map.mean().item()
        pred = "STEGO" if recon_error > threshold else "CLEAN"
    logging.info(f"Reconstruction error: {recon_error:.6f} | Prediction: {pred}")
    return recon_error, pred, x.cpu()[0], x_hat.cpu()[0], error_map.cpu()[0]

# -------------------------
# Enhanced visualization
# -------------------------
def visualize_with_info(x, x_hat, error_map, recon_error, pred, save_path):
    x_vis = x.mean(axis=0)
    xhat_vis = x_hat.mean(axis=0)
    diff = error_map.mean(axis=0)

    plt.figure(figsize=(12,4))
    
    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(x_vis, cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title("Reconstruction")
    plt.imshow(xhat_vis, cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title(f"Error Heatmap\nRecon Error: {recon_error:.6f}\nPrediction: {pred}")
    plt.imshow(diff, cmap='hot')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()
    logging.info(f"Visualization saved at: {save_path}")

# -------------------------
# Interactive image selection
# -------------------------
root = tk.Tk()
root.withdraw()
img_path = filedialog.askopenfilename(
    title="Select an image to evaluate",
    initialdir=Config.DATASET_DIR,
    filetypes=[("PNG images", "*.png"), ("All files", "*.*")]
)
if not img_path:
    logging.error("No image selected. Exiting.")
    raise SystemExit("No image selected.")
img_name = os.path.basename(img_path)
logging.info(f"Selected image: {img_name}")

# -------------------------
# Threshold from config
# -------------------------
threshold = Config.VAE_RECONTRUCTION_LOSS_THRESHOLD
logging.info(f"Using VAE reconstruction loss threshold: {threshold}")

# -------------------------
# Run inference
# -------------------------
error, pred, x, x_hat, error_map = run_inference(img_path, threshold)

# -------------------------
# Save visualization with execution count suffix
# -------------------------
vis_path = os.path.join(
    Config.EXECUTION_DIR,
    f"execution_{Config.EXECUTION_COUNT}"
)

Config.EXECUTION_COUNT += 1  # Increment for next execution

visualize_with_info(x, x_hat, error_map, error, pred, vis_path)
logging.info("Executor finished successfully.")