# src/evaluate.py

import os
import sys
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, confusion_matrix
)

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from torch.utils.data import DataLoader

# -------------------------
# Fix import paths
# -------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.config import Config
from dataset import VAEDataset
from model.vae_2 import VAE

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VAE_EVAL")


# ==========================
# Reconstruction Visualization
# ==========================
def save_reconstruction_images(x, x_hat, save_path, idx):
    x = x.cpu().numpy()[0]
    x_hat = x_hat.cpu().numpy()[0]

    # Collapse 30 channels → 2D
    x_vis = x.mean(axis=0)
    xhat_vis = x_hat.mean(axis=0)
    diff = np.abs(x_vis - xhat_vis)

    # Ensure consistent size
    H, W = Config.IMG_HEIGHT, Config.IMG_WIDTH
    if x_vis.shape != (H, W):
        x_vis = np.resize(x_vis, (H, W))
    if xhat_vis.shape != (H, W):
        xhat_vis = np.resize(xhat_vis, (H, W))
    if diff.shape != (H, W):
        diff = np.resize(diff, (H, W))

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

    file_path = os.path.join(save_path, f"reconstructed_{idx}.png")
    plt.savefig(file_path)
    plt.close()

    return file_path


# ==========================
# Main Evaluation
# ==========================
def main():

    logger.info("Starting evaluation |")

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    # -------------------------
    # Dataset (must return labels)
    # -------------------------
    dataset = VAEDataset(
        os.path.join(Config.VAE_TESTING_DATASET),
        with_labels=True
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # -------------------------
    # Load Model
    # -------------------------
    logger.info(F"Loading model checkpoint | Execution Mode {Config.DEVICE} | Model Nmae : {Config.CHECKPOINT_NAME}")
    model = VAE(input_channels=Config.IMG_CHANNELS, latent_dim=Config.LATENT_DIM)
    model.load_state_dict(
        torch.load(
            os.path.join(Config.CHECKPOINTS_DIR, Config.CHECKPOINT_NAME),
            map_location=Config.DEVICE
        )
    )
    model = model.to(Config.DEVICE)
    model.eval()

    scores = []
    labels = []
    sample_images = []

    # -------------------------
    # Inference Loop
    # -------------------------
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):

            x = x.to(Config.DEVICE)
            x_hat, _, _ = model(x)

            # Ensure reconstructed size matches input
            if x_hat.shape != x.shape:
                x_hat = F.interpolate(x_hat, size=(Config.IMG_HEIGHT, Config.IMG_WIDTH),
                                      mode='bilinear', align_corners=False)

            error = ((x - x_hat) ** 2).mean().item()

            scores.append(error)
            labels.append(y.item())

            # Save first few reconstructions
            if idx < 5:
                img_path = save_reconstruction_images(
                    x, x_hat, Config.RESULTS_DIR, idx
                )
                sample_images.append((img_path, y.item()))

    scores = np.array(scores)
    labels = np.array(labels)

    # -------------------------
    # Threshold (from CLEAN)
    # -------------------------
    clean_scores = scores[labels == 0]
    threshold = np.mean(clean_scores) + 2 * np.std(clean_scores)
    preds = (scores > threshold).astype(int)

    # -------------------------
    # Metrics
    # -------------------------
    logger.info("Calculating metrics")
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    print("\n\n")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall: {rec:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"AUC: {roc_auc:.4f}")

    # -------------------------
    # ROC Curve
    # -------------------------
    logger.info("Generating ROC curve")
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.title("ROC Curve")
    roc_path = os.path.join(Config.RESULTS_DIR, "ROC_Curve.png")
    plt.savefig(roc_path)
    plt.close()

    # -------------------------
    # Histogram
    # -------------------------
    logger.info("Generating error distribution histogram")
    stego_scores = scores[labels == 1]

    plt.figure()
    plt.hist(clean_scores, bins=50, alpha=0.5, label="Clean")
    plt.hist(stego_scores, bins=50, alpha=0.5, label="Stego")
    plt.axvline(threshold, color='r', label="Threshold")
    plt.legend()
    plt.title("Error Distribution")
    hist_path = os.path.join(Config.RESULTS_DIR, "Error_Distribution_Histogram.png")
    plt.savefig(hist_path)
    plt.close()

    # -------------------------
    # Confusion Matrix
    # -------------------------
    logger.info("Generating confusion matrix")
    cm = confusion_matrix(labels, preds)

    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    cm_path = os.path.join(Config.RESULTS_DIR, "Confusion_Matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # -------------------------
    # PDF REPORT
    # -------------------------
    logger.info("Generating report")
    pdf_path = os.path.join(Config.RESULTS_DIR, "report.pdf")
    pdf = canvas.Canvas(pdf_path, pagesize=letter)

    # Page 1: Metrics
    pdf.drawString(50, 750, "VAE Steganography Detection Report")
    pdf.drawString(50, 730, f"Accuracy: {acc:.4f}")
    pdf.drawString(50, 710, f"Precision: {prec:.4f}")
    pdf.drawString(50, 690, f"Recall: {rec:.4f}")
    pdf.drawString(50, 670, f"F1 Score: {f1:.4f}")
    pdf.drawString(50, 650, f"AUC: {roc_auc:.4f}")
    pdf.drawString(50, 630, f"Threshold: {threshold:.6f}")

    pdf.drawImage(roc_path, 50, 400, width=500, height=200)
    pdf.showPage()

    # Page 2: Histogram
    pdf.drawString(50, 750, "Error Distribution")
    pdf.drawImage(hist_path, 50, 400, width=500, height=200)
    pdf.showPage()

    # Page 3: Confusion Matrix
    pdf.drawString(50, 750, "Confusion Matrix")
    pdf.drawImage(cm_path, 50, 400, width=500, height=200)
    pdf.showPage()

    # Reconstruction Pages
    for img_path, label in sample_images:
        pdf.drawString(50, 750, "Reconstruction Analysis")
        label_text = "STEGO" if label == 1 else "CLEAN"
        pdf.drawString(50, 720, f"Sample Type: {label_text}")
        pdf.drawImage(img_path, 50, 300, width=500, height=200)
        pdf.showPage()

    pdf.save()

    logger.info(f"Evaluation complete. Report saved at: {pdf_path}")


if __name__ == "__main__":
    main()