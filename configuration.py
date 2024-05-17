COVID_DATASET = "COVID-19_Radiography_Dataset"
ROOT_DIR = r"dataset/COVID-19_Radiography_Dataset/"
EPOCHS = 30
LR = 0.003
LR = 0.003
SCHEDULER_MAX_IT = 30
WEIGH_DECAY = 1e-4
EPSILON = 1e-4

# train loop
BATCH_SIZE = 64
TEST_SIZE = 0.5
TRAIN_SIZE = 1 - TEST_SIZE
EPOCHS = 30
USE_INDEX = False
# callback
PATIENCE = 3

TOP_K_SAVES = 1
# training loop
NUM_TRIALS = 1

INDICES_DIR = "indices/"
CHECKPOINTS_DIR = "checkpoints/"
METRICS_DIR = "metrics/"
WANDB_PROJECT = "Covid-19-ml-project"

# model directories
CONVNEXT_DIR = CHECKPOINTS_DIR + "convnext/"
CONVNEXT_BILATERAL_DIR = CHECKPOINTS_DIR + "convnext_bilateral/"
MLP_DIR = CHECKPOINTS_DIR + "mlp/"

# model file names
CONVNEXT_FILENAME = "convnext_"
CONVNEXT_BILATERAL_FILENAME = "convnext_bilateral_"
MLP_FILENAME = "mlp_"

# csv file names
CONVNEXT_CSV_FILENAME = METRICS_DIR + CONVNEXT_FILENAME + "metrics.csv"
CONVNEXT_BILATERAL_CSV_FILENAME = METRICS_DIR + CONVNEXT_FILENAME + "metrics.csv"
MLP_CSV_FILENAME = METRICS_DIR + MLP_FILENAME + "metrics.csv"


# transformed images directories
MLP_FEATURES_DIR = "lbp/"
