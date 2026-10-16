from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_dataset
from config import (
    PROJECT_ROOT,
    TARGET_COLUMN,
    ID_COLUMN,
)


FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def run_eda(df):
    """Create exploratory charts and save them to outputs/figures."""

    # ID identifies a row; it is not a predictive feature.
    data = df.drop(columns=[ID_COLUMN]).copy()

    # 1. Target distribution
    target_counts = data[TARGET_COLUMN].value_counts().sort_index()

    print("\nTARGET DISTRIBUTION")
    print(target_counts)
    print("\nTarget percentages:")
    print((target_counts / len(data) * 100).round(2))

    plt.figure(figsize=(7, 5))
    sns.countplot(data=data, x=TARGET_COLUMN)
    plt.title("Credit Default Target Distribution")
    plt.xlabel("Default next month (0 = No, 1 = Yes)")
    plt.ylabel("Number of clients")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=150)
    plt.close()

    # 2. Age distribution by default status
    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=data,
        x="AGE",
        hue=TARGET_COLUMN,
        bins=25,
        element="step",
        stat="density",
        common_norm=False,
    )
    plt.title("Age Distribution by Default Status")
    plt.xlabel("Age")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "age_by_default.png", dpi=150)
    plt.close()

    # 3. Credit limit by default status
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=data, x=TARGET_COLUMN, y="LIMIT_BAL")
    plt.title("Credit Limit by Default Status")
    plt.xlabel("Default next month (0 = No, 1 = Yes)")
    plt.ylabel("Credit limit")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "credit_limit_by_default.png", dpi=150)
    plt.close()

    # 4. Default rate by repayment status in PAY_0
    default_rates = (
        data.groupby("PAY_0", observed=True)[TARGET_COLUMN]
        .mean()
        .mul(100)
        .rename("default_rate_percent")
        .reset_index()
    )

    print("\nDEFAULT RATE BY PAY_0")
    print(default_rates.round(2).to_string(index=False))

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=default_rates,
        x="PAY_0",
        y="default_rate_percent",
    )
    plt.title("Default Rate by Most Recent Repayment Status")
    plt.xlabel("PAY_0 repayment-status code")
    plt.ylabel("Clients who defaulted (%)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "default_rate_by_pay0.png", dpi=150)
    plt.close()

    # 5. Correlation heatmap for numeric variables
    plt.figure(figsize=(16, 12))
    correlation = data.corr(numeric_only=True)
    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        linewidths=0.1,
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    print("\nEDA completed.")
    print(f"Charts saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    dataset = load_dataset()
    run_eda(dataset)