
from pathlib import Path
import json
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from data_loader import load_dataset
from config import PROJECT_ROOT, TARGET_COLUMN, ID_COLUMN, RANDOM_STATE


METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def train_and_evaluate():
    # Load the dataset
    df = load_dataset()

    # Separate prediction target from input features
    X = df.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    y = df[TARGET_COLUMN]

    # Keep the original class proportions in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Scale features and train Logistic Regression
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    model.fit(X_train, y_train)

    # Evaluate only on the held-out test set
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(
            y_test, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_test, predictions, zero_division=0
        )),
        "f1_score": float(f1_score(
            y_test, predictions, zero_division=0
        )),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }

    print("\n" + "=" * 55)
    print("LOGISTIC REGRESSION — TEST SET RESULTS")
    print("=" * 55)

    for metric, value in metrics.items():
        print(f"{metric.replace('_', ' ').title():<15}: {value:.4f}")

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_test,
        predictions,
        target_names=["No Default (0)", "Default (1)"],
        zero_division=0,
    ))

    print("CONFUSION MATRIX")
    print("Rows = actual class; columns = predicted class")
    print(confusion_matrix(y_test, predictions))

    # Save metrics for reproducibility
    metrics_path = METRICS_DIR / "logistic_regression_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows:  {len(X_test):,}")


if __name__ == "__main__":
    train_and_evaluate()
