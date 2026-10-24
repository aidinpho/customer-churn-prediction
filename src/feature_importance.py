
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "iranian_churn.csv"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=42,
        stratify=y_train_val,
    )

    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train_full, y_train_full)

    importance_df = pd.DataFrame({
        "Feature": X_train_full.columns,
        "Importance": model.feature_importances_,
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False,
    ).reset_index(drop=True)

    print("\nRandom Forest — Feature Importance")
    print("=" * 50)
    print(
        importance_df.to_string(
            index=False,
            formatters={"Importance": "{:.4f}".format},
        )
    )

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=True,
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"],
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()

    output_dir = PROJECT_ROOT / "screenshots"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "feature-importance.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nFeature importance chart saved to: {output_path}")


if __name__ == "__main__":
    main()
