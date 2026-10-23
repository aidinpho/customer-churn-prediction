
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "iranian_churn.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    # Standardize column names
    df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)

    print("Dataset shape:", df.shape)

    print("\nTarget distribution:")
    print(df["Churn"].value_counts().sort_index())

    print("\nTarget distribution (%):")
    print(
        (df["Churn"].value_counts(normalize=True).sort_index() * 100)
        .round(2)
    )

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nStatistical summary:")
    print(df.describe().round(2).T)

    print("\nUnique values in selected columns:")
    for column in ["Complains", "Age Group", "Tariff Plan", "Status", "Churn"]:
        print(f"{column}: {sorted(df[column].unique())}")

    
    print("\nChurn distribution among duplicate rows:")
    duplicate_rows = df[df.duplicated(keep=False)]

    print(
        duplicate_rows["Churn"]
        .value_counts()
        .sort_index()
    )

    print("\nChurn values within identical feature groups:")
    feature_columns = df.columns.drop("Churn")

    grouped_churn = (
        df.groupby(list(feature_columns), dropna=False)["Churn"]
        .agg(["size", "nunique"])
    )

    print("Identical feature groups:", len(grouped_churn))
    print(
        "Groups with conflicting Churn labels:",
        (grouped_churn["nunique"] > 1).sum()
    )

    
    print("\nExamples of conflicting Churn labels:")

    feature_columns = df.columns.drop("Churn")

    conflicting_groups = (
        df.groupby(list(feature_columns), dropna=False)["Churn"]
        .nunique()
    )

    conflicting_groups = conflicting_groups[
        conflicting_groups > 1
    ]

    for features in conflicting_groups.index[:5]:
        mask = pd.Series(True, index=df.index)

        for column, value in zip(feature_columns, features):
            mask &= df[column].eq(value)

        print(df.loc[mask].to_string(index=False))
        print("-" * 60)

    
    print("\nDuplicate handling comparison:")

    without_exact_duplicates = df.drop_duplicates()

    without_duplicate_features = df.drop_duplicates(
        subset=df.columns.drop("Churn")
    )

    datasets = {
        "Original": df,
        "Without exact duplicates": without_exact_duplicates,
        "One row per feature combination": without_duplicate_features,
    }

    for name, data in datasets.items():
        print(f"\n{name}")
        print("Rows:", len(data))
        print("Churn counts:")
        print(data["Churn"].value_counts().sort_index().to_dict())
        print(
            "Churn rate (%):",
            round(data["Churn"].mean() * 100, 2),
        )

    
    print("\nFeature averages by Churn:")
    print(
        df.groupby("Churn")
        .mean(numeric_only=True)
        .T.round(2)
    )

    print("\nChurn rate by selected categorical features:")

    for column in ["Complains", "Tariff Plan", "Status", "Age Group"]:
        print(f"\n{column}")

        churn_rate = (
            df.groupby(column)["Churn"]
            .agg(["mean", "count"])
        )

        churn_rate["mean"] *= 100
        churn_rate = churn_rate.rename(
            columns={"mean": "Churn rate (%)"}
        )

        print(churn_rate.round(2))



if __name__ == "__main__":
    main()
