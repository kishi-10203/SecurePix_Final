# model/vae_2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# -------------------------
# Fix import paths
# -------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


from src.config import Config






class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # -----------------------
        # Convolutional Encoder
        # -----------------------
        self.conv_layers = nn.Sequential(
            nn.Conv2d(Config.IMG_CHANNELS, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        # -----------------------
        # Compute dynamic flattened size
        # -----------------------
        self._conv_output_size = self._get_conv_output_size()
        
        # -----------------------
        # Linear layers for latent vectors
        # -----------------------
        self.fc_mu = nn.Linear(self._conv_output_size, Config.LATENT_DIM)
        self.fc_logvar = nn.Linear(self._conv_output_size, Config.LATENT_DIM)

        # -----------------------
        # Decoder
        # -----------------------
        self.decoder_input = nn.Linear(Config.LATENT_DIM, self._conv_output_size)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, Config.IMG_CHANNELS, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Because residuals are in [-1,1]
        )

    # -----------------------
    # Helper: compute conv output size dynamically
    # -----------------------
    def _get_conv_output_size(self):
        dummy = torch.zeros(1, Config.IMG_CHANNELS, Config.IMG_HEIGHT, Config.IMG_WIDTH)
        out = self.conv_layers(dummy)
        return int(np.prod(out.size()))

    # -----------------------
    # Encoder
    # -----------------------
    def encode(self, x):
        x = self.conv_layers(x)
        x_flat = x.view(x.size(0), -1)
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        return mu, logvar

    # -----------------------
    # Reparameterization
    # -----------------------
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # -----------------------
    # Decoder
    # -----------------------
    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), 256, 
                   Config.IMG_HEIGHT // 8, Config.IMG_WIDTH // 8)  # match conv downsampling
        x = self.deconv_layers(x)
        return x

    # -----------------------
    # Forward pass
    # -----------------------
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar