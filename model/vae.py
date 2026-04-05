# model/vae.py
import os
import sys
import torch
import torch.nn as nn

# -------------------------
# Fix import paths
# -------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.config import Config

cfg = Config()

class VAE(nn.Module):
    def __init__(self, latent_dim=cfg.LATENT_DIM, input_channels=cfg.IMG_CHANNELS, device=cfg.DEVICE):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.device = device

        # --------------------
        # Encoder
        # --------------------
        # Input: [batch, 30, 50, 200]
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # 50x200 -> 25x100
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),             # 25x100 -> 13x50
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),            # 13x50 -> 7x25
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Flatten features for latent space
        self.flatten = nn.Flatten()
        conv_output_size = 256 * 7 * 25  # channels * H * W after conv stack
        self.fc_mu = nn.Linear(conv_output_size, latent_dim)
        self.fc_logvar = nn.Linear(conv_output_size, latent_dim)

        # --------------------
        # Decoder
        # --------------------
        self.fc_dec = nn.Linear(latent_dim, conv_output_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 7x25 -> 14x50
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 14x50 -> 28x100
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),  # 28x100 -> 56x200
            nn.Tanh()  # Output normalized [-1,1]
        )

        # Move model to device
        self.to(self.device)

    # --------------------
    # Encoder → Latent → Decoder
    # --------------------
    def encode(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_dec(z)
        x = x.view(-1, 256, 7, 25)
        x = self.decoder(x)
        # crop to exact height=50 if needed
        x = x[:, :, :cfg.IMG_HEIGHT, :cfg.IMG_WIDTH]
        return x

    def forward(self, x):
        x = x.to(self.device)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    # --------------------
    # Convenience functions
    # --------------------
    def infer(self, x):
        self.eval()
        x = x.to(self.device)
        with torch.no_grad():
            x_hat, mu, logvar = self.forward(x)
        return x_hat, mu, logvar

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()