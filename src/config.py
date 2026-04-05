# src/config.py
import os
import torch

class Config:
    # ==========================
    # Dataset paths
    # ==========================
    #  project root 
    ROOT_DIR = r"D:\Final_Year_Project\SecurePix_Final"

    # Git - Untracked Dataset Directory
    DATASET_DIR =  r"D:\Final_Year_Project\SecurePix_Dataset"

    #Training\Testing|Validation Dataset files - .txt
    VAE_TRAINING_DATASET = r"D:\Final_Year_Project\SecurePix_Final\data\train.txt"
    VAE_TESTING_DATASET = r"D:\Final_Year_Project\SecurePix_Final\data\test.txt"
    VAE_VALIDATION_DATASET = r"D:\Final_Year_Project\SecurePix_Final\data\val.txt"


    # Clean img - VAE feed Directory
    SOURCE_DATA_DIR = r"D:\Final_Year_Project\SecurePix_Dataset\CLEAN_IMGAES"

    # Model -checkpoint Directory
    CHECKPOINTS_DIR = r"D:\Final_Year_Project\SecurePix_Final\checkpoints"

    #Logs Directory
    LOGS_DIR = os.path.join(ROOT_DIR, "logs")

    #Results Directory
    RESULTS_DIR = r"D:\Final_Year_Project\SecurePix_Final\results"

    #Execution Output Directory
    EXECUTION_DIR = r"D:\Final_Year_Project\SecurePix_Final\results\execution_output"

    # ==========================
    # VAE settin"D:\Final_Year_Project\SecurePix_Final\results\execution_output"    # ==========================
    IMG_CHANNELS = 30
    IMG_HEIGHT = 50
    IMG_WIDTH = 200
    LATENT_DIM = 64

    # ==========================
    # Training parameters
    # ==========================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 2 #50 - reduced for testing
    BETA = 1.0
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED = 42

    # ==========================
    # Checkpoint
    # ==========================
    CHECKPOINT_NAME = "vae_2.pth"

    # ==========================
    # Logging
    # ==========================
    LOG_LEVEL = "INFO"



    # VAE Threshold for anomaly detection  - calculated from evaluation 
    VAE_RECONTRUCTION_LOSS_THRESHOLD = 0.471960

    # EXECUTION - Counter 
    EXECUTION_COUNT = 1

    # ==========================
    # Ensure directories exist
    # ==========================
    @classmethod
    def ensure_dirs(cls):
        for path in [cls.CHECKPOINTS_DIR, cls.LOGS_DIR, cls.RESULTS_DIR]:
            os.makedirs(path, exist_ok=True)