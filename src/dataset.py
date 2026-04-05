# src/dataset.py

from torch.utils.data import Dataset
import torch
import os
from utils.srm_filters import extract_residual


class VAEDataset(Dataset):
    def __init__(self, file_list, with_labels=False):
        self.files = [line.strip() for line in open(file_list)]
        self.with_labels = with_labels

        # 🔥 Base directory where your dataset actually lives
        self.base_dir = r"E:\SecurePix_Dataset"

    def __len__(self):
        return len(self.files)

    def _resolve_path(self, path):
        """
        Convert relative paths → absolute paths
        """
        # If already absolute, return as-is
        if os.path.isabs(path):
            return path

        # If path starts with SecurePix_Dataset, fix it
        if path.startswith("SecurePix_Dataset"):
            # Remove leading folder name and attach correct base
            relative_part = path.replace("SecurePix_Dataset\\", "")
            full_path = os.path.join(self.base_dir, relative_part)
        else:
            # Fallback (still attach base_dir)
            full_path = os.path.join(self.base_dir, path)

        return full_path

    def __getitem__(self, idx):
        path = self.files[idx]

        # Fix path here
        full_path = self._resolve_path(path)

        # Fail fast if path is wrong
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")

        # Extract residual (30 channels)
        residual = extract_residual(full_path)
        residual = torch.tensor(residual, dtype=torch.float32)

        # If labels needed (evaluation mode)
        if self.with_labels:
            if "STEG" in full_path.upper():
                label = 1   # stego
            else:
                label = 0   # clean

            return residual, label

        return residual