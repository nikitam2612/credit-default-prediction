
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from data_loader import load_dataset
from config import PROJECT_ROOT, TARGET_COLUMN, ID_COLUMN, RANDOM_STATE


METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(y_true, predictions, probabilities):
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def main():
    df = load_dataset()

    # Exclude the row identifier and target from model inputs.
    X = df.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    y = df[TARGET_COLUMN]

    # First, hold out 20% for the final test set.
    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Split the remaining 80% into training and validation sets.
    # Overall proportions: 60% train, 20% validation, 20% test.
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_development,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("Training Random Forest...")
    model.fit(X_train, y_train)

    # Choose the threshold using validation data only.
    validation_probabilities = model.predict_proba(X_validation)[:, 1]

    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    threshold_results = []

    for threshold in thresholds:
        validation_predictions = (
            validation_probabilities >= threshold
        ).astype(int)

        threshold_results.append({
            "threshold": threshold,
            "precision": precision_score(
                y_validation,
                validation_predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y_validation,
                validation_predictions,
                zero_division=0,
            ),
            "f1_score": f1_score(
                y_validation,
                validation_predictions,
                zero_division=0,
            ),
        })

    results_df = pd.DataFrame(threshold_results)

    print("\nVALIDATION RESULTS BY THRESHOLD")
    print(results_df.round(4).to_string(index=False))

    # Select the threshold with the highest validation F1-score.
    # If tied, pandas selects the first threshold in the table.
    best_row = results_df.loc[results_df["f1_score"].idxmax()]
    best_threshold = float(best_row["threshold"])

    print(f"\nSelected threshold: {best_threshold:.2f}")
    print("Selection criterion: highest validation F1-score")

    # Evaluate the selected threshold once on the untouched test set.
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= best_threshold).astype(int)

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
        test_probabilities,
    )

    print("\nFINAL TEST RESULTS AT SELECTED THRESHOLD")
    for metric, value in test_metrics.items():
        print(f"{metric.replace('_', ' ').title():<15}: {value:.4f}")

    print("\nCONFUSION MATRIX")
    print("Rows = actual class; columns = predicted class")
    print(confusion_matrix(y_test, test_predictions))

    # Save validation results and final test results separately.
    results_path = METRICS_DIR / "threshold_tuning_results.json"

    output = {
        "model": "Random Forest",
        "threshold_selection": "Highest validation F1-score",
        "selected_threshold": best_threshold,
        "validation_results": threshold_results,
        "test_metrics": test_metrics,
        "test_confusion_matrix": confusion_matrix(
            y_test, test_predictions
        ).tolist(),
        "split_sizes": {
            "training": len(X_train),
            "validation": len(X_validation),
            "test": len(X_test),
        },
    }

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print(f"\nResults saved to: {results_path}")
    print(
        f"Rows — train: {len(X_train):,}, "
        f"validation: {len(X_validation):,}, "
        f"test: {len(X_test):,}"
    )


if __name__ == "__main__":
    main()
