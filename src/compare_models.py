
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from data_loader import load_dataset
from config import PROJECT_ROOT, TARGET_COLUMN, ID_COLUMN, RANDOM_STATE


METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def main():
    df = load_dataset()

    X = df.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    random_forest_model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    results = [
        evaluate_model(
            "Logistic Regression",
            logistic_model,
            X_train, X_test, y_train, y_test,
        ),
        evaluate_model(
            "Random Forest",
            random_forest_model,
            X_train, X_test, y_train, y_test,
        ),
    ]

    results_df = pd.DataFrame(results).set_index("model")
    print("\nMODEL COMPARISON — TEST SET")
    print(results_df.round(4).to_string())

    output_path = METRICS_DIR / "model_comparison.csv"
    results_df.to_csv(output_path)

    print(f"\nComparison saved to: {output_path}")


if __name__ == "__main__":
    main()
