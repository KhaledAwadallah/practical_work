import torch

# Data Path
DATA_PATH = "data/muv.csv"

# Model hyperparameter
# Frequent Hitters (Neural Network)
INPUT_SIZE = 2248
HS1 = 64
HS2 = 32
HS3 = 16
OUTPUT_SIZE = 1
NUM_EPOCHS = 5
BATCH_SIZE = 8192
LEARNING_RATE = 0.01

# Experiment Settings
SEEDS = [0, 1, 2, 3, 4]
TRAIN_TASKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
VAL_TASKS = [10, 11, 12]
TEST_TASKS = [13, 14, 15]
RF_TASKS = [13, 14, 15]

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

