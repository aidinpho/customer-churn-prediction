
from pathlib import Path
import math
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "threshold.txt"

FEATURES = [
    "Call Failure",
    "Complains",
    "Subscription Length",
    "Charge Amount",
    "Seconds of Use",
    "Frequency of use",
    "Frequency of SMS",
    "Distinct Called Numbers",
    "Age Group",
    "Tariff Plan",
    "Age",
    "Customer Value",
]

CATEGORICAL_VALUES = {
    "Complains": {0, 1},
    "Age Group": {1, 2, 3, 4, 5},
    "Tariff Plan": {1, 2},
}

NON_NEGATIVE_FEATURES = [
    "Call Failure",
    "Subscription Length",
    "Charge Amount",
    "Seconds of Use",
    "Frequency of use",
    "Frequency of SMS",
    "Distinct Called Numbers",
    "Customer Value",
]

INTEGER_FEATURES = {
            "Call Failure",
            "Subscription Length",
            "Seconds of Use",
            "Frequency of use",
            "Frequency of SMS",
            "Distinct Called Numbers",
            "Age Group",
            "Tariff Plan",
}


def validate_customer_data(customer_data):
    if not isinstance(customer_data, dict):
        raise TypeError("Customer data must be a dictionary.")

    missing_features = [
        feature for feature in FEATURES
        if feature not in customer_data
    ]

    unexpected_features = [
        feature for feature in customer_data
        if feature not in FEATURES
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    if unexpected_features:
        raise ValueError(
            f"Unexpected features: {unexpected_features}"
        )

    for feature in FEATURES:
        value = customer_data[feature]

        if isinstance(value, bool) or not isinstance(
            value, (int, float)
        ):
            raise TypeError(
                f"'{feature}' must be a number."
            )

        if not math.isfinite(value):
            raise ValueError(
                f"'{feature}' must be a finite number."
        )

        

        if feature in INTEGER_FEATURES and value != int(value):
            raise ValueError(
                f"'{feature}' must be a whole number."
            )


        if feature in CATEGORICAL_VALUES:
            if value not in CATEGORICAL_VALUES[feature]:
                raise ValueError(
                    f"'{feature}' must be one of "
                    f"{sorted(CATEGORICAL_VALUES[feature])}."
                )

        if feature in NON_NEGATIVE_FEATURES and value < 0:
            raise ValueError(
                f"'{feature}' cannot be negative."
            )

        if feature == "Age" and not 15 <= value <= 55:
            raise ValueError(
                "'Age' must be between 15 and 55 "
                "based on the training dataset."
            )


def predict_churn(customer_data):
    validate_customer_data(customer_data)

    customer = pd.DataFrame([customer_data], columns=FEATURES)

    model = joblib.load(MODEL_PATH)
    threshold = float(
        THRESHOLD_PATH.read_text(encoding="utf-8").strip()
    )

    probability = model.predict_proba(customer)[0, 1]
    prediction = int(probability >= threshold)

    return {
        "churn_prediction": prediction,
        "churn_probability": float(probability),
        "threshold": threshold,
    }



def main():
    # Example customer data for testing only.
    example_customer = {
        "Call Failure": 8,
        "Complains": 0,
        "Subscription Length": 30,
        "Charge Amount": 2,
        "Seconds of Use": 2500,
        "Frequency of use": 40,
        "Frequency of SMS": 20,
        "Distinct Called Numbers": 15,
        "Age Group": 3,
        "Tariff Plan": 1,
        "Age": 30,
        "Customer Value": 200,
    }

    # Test 1: Missing feature
    invalid_customer = example_customer.copy()
    invalid_customer.pop("Age")

    try:
        predict_churn(invalid_customer)
    except ValueError as error:
        print("Missing feature test:", error)

    # Test 2: Non-numeric value
    invalid_customer = example_customer.copy()
    invalid_customer["Age"] = "thirty"

    try:
        predict_churn(invalid_customer)
    except TypeError as error:
        print("Non-numeric test:", error)


    # Test 3: NaN value
    invalid_customer = example_customer.copy()
    invalid_customer["Age"] = float("nan")

    try:
        predict_churn(invalid_customer)
    except ValueError as error:
        print("NaN test:", error)

    # Test 4: Infinite value
    invalid_customer = example_customer.copy()
    invalid_customer["Age"] = float("inf")

    try:
        predict_churn(invalid_customer)
    except ValueError as error:
        print("Infinity test:", error)

    # Test 5: Invalid categorical value
    invalid_customer = example_customer.copy()
    invalid_customer["Complains"] = 2

    try:
        predict_churn(invalid_customer)
    except ValueError as error:
        print("Invalid Complains test:", error)

    # Test 6: Negative age
    invalid_customer = example_customer.copy()
    invalid_customer["Age"] = -5

    try:
        predict_churn(invalid_customer)
    except ValueError as error:
        print("Invalid Age test:", error)

    # Test 7: Fractional count
    invalid_customer = example_customer.copy()
    invalid_customer["Frequency of SMS"] = 20.5

    try:
        predict_churn(invalid_customer)
    except ValueError as error:
        print("Fractional count test:", error)

    result = predict_churn(example_customer)

    print("Churn Prediction")
    print("-" * 30)
    print(f"Prediction: {result['churn_prediction']}")
    print(f"Churn probability: {result['churn_probability']:.2%}")
    print(f"Decision threshold: {result['threshold']:.2f}")

    if result["churn_prediction"] == 1:
        print("Result: Customer is flagged as at risk of churn.")
    else:
        print("Result: Customer is not flagged as at risk of churn.")


if __name__ == "__main__":
    main()
