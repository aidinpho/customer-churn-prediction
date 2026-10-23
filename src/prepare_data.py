
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "iranian_churn.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Standardize column names.
    df.columns = df.columns.str.strip().str.replace(
        r"\s+", " ", regex=True
    )

    return df


def main():
    from sklearn.model_selection import train_test_split

    df = load_data()

    target = "Churn"
    excluded_features = ["Status"]

    X = df.drop(columns=[target, *excluded_features])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training features shape:", X_train.shape)
    print("Test features shape:", X_test.shape)

    print("\nTraining target distribution (%):")
    print((y_train.value_counts(normalize=True) * 100).round(2))

    print("\nTest target distribution (%):")
    print((y_test.value_counts(normalize=True) * 100).round(2))


if __name__ == "__main__":
    main()
