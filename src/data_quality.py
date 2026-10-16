from data_loader import load_dataset
from config import TARGET_COLUMN, ID_COLUMN


def inspect_data_quality(df):
    """Report data quality issues without changing the original data."""

    print("\n" + "=" * 60)
    print("1. DATASET SHAPE")
    print("=" * 60)
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\n" + "=" * 60)
    print("2. DATA TYPES")
    print("=" * 60)
    print(df.dtypes.to_string())

    print("\n" + "=" * 60)
    print("3. MISSING VALUES")
    print("=" * 60)
    missing = df.isnull().sum()
    print(missing[missing > 0].to_string() if missing.any() else "No missing values found.")

    print("\n" + "=" * 60)
    print("4. DUPLICATE ROWS")
    print("=" * 60)
    print(f"Exact duplicate rows: {df.duplicated().sum():,}")
    print(f"Duplicate IDs: {df[ID_COLUMN].duplicated().sum():,}")

    print("\n" + "=" * 60)
    print("5. TARGET VALIDATION")
    print("=" * 60)
    print("Target values:")
    print(df[TARGET_COLUMN].value_counts(dropna=False).sort_index().to_string())
    print("\nUnexpected target values:")
    print(df.loc[~df[TARGET_COLUMN].isin([0, 1]), TARGET_COLUMN].value_counts().to_string()
          or "None")

    print("\n" + "=" * 60)
    print("6. CATEGORICAL CODE COUNTS")
    print("=" * 60)
    for column in ["SEX", "EDUCATION", "MARRIAGE",
                   "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False).sort_index().to_string())

    print("\n" + "=" * 60)
    print("7. NUMERIC SUMMARY")
    print("=" * 60)
    print(df.describe().round(2).to_string())


if __name__ == "__main__":
    data = load_dataset()
    inspect_data_quality(data)