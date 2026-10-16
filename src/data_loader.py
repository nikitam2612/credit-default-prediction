import pandas as pd

from config import DATASET_PATH, TARGET_COLUMN, ID_COLUMN


def load_dataset():
    """Load the UCI credit default dataset from the Excel file."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}\n"
            "Please place the Excel file inside data/raw/."
        )

    # The first row in this Excel file contains the actual column names.
    df = pd.read_excel(DATASET_PATH, header=1)

    # Remove accidental spaces from column names.
    df.columns = df.columns.astype(str).str.strip()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found. "
            f"Columns found: {df.columns.tolist()}"
        )

    if ID_COLUMN not in df.columns:
        raise ValueError(
            f"Expected ID column '{ID_COLUMN}' was not found."
        )

    return df


if __name__ == "__main__":
    df = load_dataset()

    print("\nDataset loaded successfully!")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    print("\nFirst five rows:")
    print(df.head())
    print("\nTarget distribution:")
    print(df[TARGET_COLUMN].value_counts(dropna=False).sort_index())