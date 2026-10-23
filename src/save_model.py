
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "iranian_churn.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "threshold.txt"


def main():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)

    X = df.drop(columns=["Churn", "Status"])
    y = df["Churn"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train_val, y_train_val)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    THRESHOLD_PATH.write_text("0.5\n", encoding="utf-8")

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Threshold saved to: {THRESHOLD_PATH}")


if __name__ == "__main__":
    main()
