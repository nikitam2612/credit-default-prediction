from pathlib import Path

# Project root folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset location
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_FILENAME = "default of credit card clients.xls"
DATASET_PATH = RAW_DATA_DIR / DATASET_FILENAME

# Reproducibility
RANDOM_STATE = 42

# Target column: whether a client defaults next month
TARGET_COLUMN = "default payment next month"

# Column that identifies a row but should not be used for prediction
ID_COLUMN = "ID"