import os

# === Configuration ===
EPOCHS = 30          # Number of training epochs
LAYERS = 1           # Number of RandomLayers in the quantum circuit
N_TRAIN = 500        # Number of training samples
N_TEST = 500         # Number of test samples
SAVE_PATH = "model/"  # Directory for saving data and models
os.makedirs(SAVE_PATH, exist_ok=True)

PREPROCESS = input("Preprocess data? (y/n): ").strip().lower() == 'y'