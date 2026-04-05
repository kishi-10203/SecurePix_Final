# utils/srm_filters.py

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from src.config import Config

# ===========================
# SRM30 filters
# ===========================
def get_srm30_filters():
    """
    Returns SRM30 filters as torch.Tensor [30,1,5,5]
    Simplified for small project implementation.
    """
    kernels = []

    # Example high-pass kernels (3x3) padded to 5x5
    base_kernels = [
        np.array([[0, 0, 0],
                  [0, 1, -1],
                  [0, 0, 0]]),
        np.array([[0, 1, 0],
                  [0, -1, 0],
                  [0, 0, 0]]),
        np.array([[1, -2, 1],
                  [-2, 4, -2],
                  [1, -2, 1]]),
        np.array([[0, 0, 0],
                  [1, -1, 0],
                  [0, 0, 0]])
    ]

    # Repeat kernels to reach 30 filters
    while len(base_kernels) < 30:
        base_kernels.append(base_kernels[len(base_kernels) % 4])

    for k in base_kernels[:30]:
        padded = np.pad(k, ((1, 1), (1, 1)), mode='constant')
        kernels.append(padded)

    # Convert to tensor [30,1,5,5]
    filters_tensor = torch.tensor(kernels, dtype=torch.float32).unsqueeze(1)
    return filters_tensor


# ===========================
# Residual extraction
# ===========================
def extract_residual(image_path):
    """
    Extract 30-channel SRM residual for given image.
    Returns tensor [30, H, W] normalized to [-1,1].
    Resizes image to Config.IMG_HEIGHT x Config.IMG_WIDTH if needed.
    """
    # Load grayscale
    img = Image.open(image_path).convert("L")
    img = img.resize((Config.IMG_WIDTH, Config.IMG_HEIGHT))
    img_np = np.array(img, dtype=np.float32) / 255.0  # normalize 0-1
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    # SRM30 filters
    filters = get_srm30_filters()  # [30,1,5,5]

    # Convolve
    residual = F.conv2d(img_tensor, filters, padding=2)  # [1,30,H,W]
    residual = residual.squeeze(0)  # [30,H,W]

    # Clip to [-1,1]
    residual = torch.clamp(residual, -1.0, 1.0)

    return residual


# ===========================
# Optional: patch extraction
# ===========================
def extract_residual_patches(image_path, patch_size=32, stride=16):
    """
    Returns overlapping patches [num_patches, 30, patch_size, patch_size]
    """
    residual = extract_residual(image_path)
    c, h, w = residual.shape

    patches = residual.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    patches = patches.contiguous().view(c, -1, patch_size, patch_size)  # [C,num_patches,patch,patch]
    patches = patches.permute(1, 0, 2, 3)  # [num_patches,C,patch,patch]

    return patches