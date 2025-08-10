import torch

# Data Paths
DATA_PATH = "../data/muv.csv"

# Model Hyperparameters
# Frequent Hitters (Neural Network)
FH_INPUT_SIZE = 2248
FH_HS1 = 64
FH_HS2 = 32
FH_HS3 = 16
FH_OUTPUT_SIZE = 1
FH_NUM_EPOCHS = 5
FH_BATCH_SIZE = 8192
FH_LEARNING_RATE = 0.01

# Experiment settings
SEEDS = [0, 1, 2, 3, 4]
TRAIN_TASKS_SLICE = slice(0, 10)
VALIDATION_TASKS_SLICE = slice(10, 13)
TEST_TASKS_SLICE = slice(13, 16)
RF_TASKS = [13, 14, 15]

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
