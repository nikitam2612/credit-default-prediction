
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from data_loader import load_dataset
from config import PROJECT_ROOT, TARGET_COLUMN, ID_COLUMN, RANDOM_STATE


FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = load_dataset()

    X = df.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    y = df[TARGET_COLUMN]

    # Match the 60% training / 20% validation / 20% test setup.
    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

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

    model.fit(X_train, y_train)

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nTOP 15 RANDOM FOREST FEATURES")
    print(importance_df.head(15).round(4).to_string(index=False))

    csv_path = METRICS_DIR / "feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)

    top_features = importance_df.head(15).sort_values("importance")

    plt.figure(figsize=(10, 7))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("Top 15 Random Forest Feature Importances")
    plt.xlabel("Importance score")
    plt.ylabel("Feature")
    plt.tight_layout()

    chart_path = FIGURES_DIR / "feature_importance.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    print(f"\nFeature table saved to: {csv_path}")
    print(f"Feature chart saved to: {chart_path}")


if __name__ == "__main__":
    main()
