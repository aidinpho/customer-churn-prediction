
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "iranian_churn.csv"


def load_data():
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

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_on_validation(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    val_proba = model.predict_proba(X_val)[:, 1]

    best_result = None

    for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]:
        val_pred = (val_proba >= threshold).astype(int)

        precision = precision_score(
            y_val, val_pred, zero_division=0
        )
        recall = recall_score(
            y_val, val_pred, zero_division=0
        )
        f1 = f1_score(
            y_val, val_pred, zero_division=0
        )

        if best_result is None or f1 > best_result["f1"]:
            best_result = {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

    return best_result



def evaluate_on_test(
    model,
    threshold,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
):
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    model.fit(X_train_full, y_train_full)

    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)

    print(f"\nThreshold: {threshold}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_pred))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            test_pred,
            zero_division=0,
        )
    )


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
        ),
        "KNN (k=5)": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []

    for model_name, model in models.items():
        result = evaluate_on_validation(
            model, X_train, y_train, X_val, y_val
        )

        results.append({
            "Model": model_name,
            "Threshold": result["threshold"],
            "Precision": result["precision"],
            "Recall": result["recall"],
            "F1": result["f1"],
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="F1", ascending=False
    ).reset_index(drop=True)

    print("\nModel Comparison — Validation Results")
    print("=" * 75)
    print(results_df.to_string(
        index=False,
        formatters={
            "Threshold": "{:.1f}".format,
            "Precision": "{:.3f}".format,
            "Recall": "{:.3f}".format,
            "F1": "{:.3f}".format,
        },
    ))


    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    best_knn = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5)),
    ])

    evaluate_on_test(
        best_knn,
        threshold=0.3,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_val=X_val,
        y_val=y_val,
    )

    best_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("\n" + "-" * 60)
    print("Random Forest")

    evaluate_on_test(
        best_rf,
        threshold=0.5,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_val=X_val,
        y_val=y_val,
    )




if __name__ == "__main__":
    main()
