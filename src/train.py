# src/train.py
import os, sys, logging
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

# Add project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.config import Config
from dataset import VAEDataset
from model.vae_2 import VAE

# Setup logging
logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger("VAE_Train")

def main():
    # Ensure dirs
    Config.ensure_dirs()
    logger.info("Initializing training...")

    # Dataset
    logger.info("Loading dataset...")
    train_dataset = VAEDataset("E:\\SecurePix\\data\\train.txt")
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # Model
    logger.info("Building model...")
    model = VAE(input_channels=train_dataset[0].shape[0], latent_dim=Config.LATENT_DIM)
    model = model.to(Config.DEVICE)

    # Optimizer
    logger.info("Building optimizer...")
    optimizer = Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Training loop
    logger.info("Starting training...")
    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(Config.DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss, recon_loss, kld = model.loss_function(recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Epoch {epoch+1}/{Config.EPOCHS}, Loss: {total_loss/len(train_loader)}")

    # Save checkpoint
    logger.info("Saving model checkpoint...")
    ckpt_path = os.path.join(Config.CHECKPOINTS_DIR, Config.CHECKPOINT_NAME)
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Training complete. Model saved at {ckpt_path}")

if __name__ == "__main__":
    main()