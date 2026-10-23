from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.predict_churn import (
    FEATURES,
    predict_churn,
    validate_customer_data,
)



def test_missing_feature():
    customer = {
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
        # Age is intentionally missing
        "Customer Value": 200,
    }

    with pytest.raises(ValueError, match="Missing features"):
        validate_customer_data(customer)


def make_valid_customer():
    return {
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


def test_non_numeric_value():
    customer = make_valid_customer()
    customer["Age"] = "thirty"

    with pytest.raises(TypeError, match="must be a number"):
        validate_customer_data(customer)


def test_nan_value():
    customer = make_valid_customer()
    customer["Age"] = float("nan")

    with pytest.raises(ValueError, match="finite number"):
        validate_customer_data(customer)


def test_infinite_value():
    customer = make_valid_customer()
    customer["Age"] = float("inf")

    with pytest.raises(ValueError, match="finite number"):
        validate_customer_data(customer)


def test_invalid_categorical_value():
    customer = make_valid_customer()
    customer["Complains"] = 2

    with pytest.raises(ValueError, match="must be one of"):
        validate_customer_data(customer)


def test_negative_customer_value():
    customer = make_valid_customer()
    customer["Customer Value"] = -10

    with pytest.raises(ValueError, match="cannot be negative"):
        validate_customer_data(customer)


def test_invalid_age():
    customer = make_valid_customer()
    customer["Age"] = -5

    with pytest.raises(ValueError, match="Age"):
        validate_customer_data(customer)


def test_fractional_count():
    customer = make_valid_customer()
    customer["Frequency of SMS"] = 20.5

    with pytest.raises(ValueError, match="whole number"):
        validate_customer_data(customer)


def test_valid_customer():
    customer = make_valid_customer()

    # Valid input should not raise an exception.
    validate_customer_data(customer)


def test_unexpected_feature():
    customer = make_valid_customer()
    customer["Unknown Feature"] = 123

    with pytest.raises(ValueError, match="Unexpected features"):
        validate_customer_data(customer)


@pytest.mark.parametrize(
    "feature, value",
    [
        ("Complains", 2),
        ("Age Group", 6),
        ("Tariff Plan", 3),
    ],
)
def test_invalid_categorical_values(feature, value):
    customer = make_valid_customer()
    customer[feature] = value

    with pytest.raises(ValueError):
        validate_customer_data(customer)


@patch("src.predict_churn.THRESHOLD_PATH")
@patch("src.predict_churn.joblib.load")
def test_predict_churn(mock_load, mock_threshold_path):
    mock_model = Mock()
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
    mock_load.return_value = mock_model

    mock_threshold_path.read_text.return_value = "0.5"

    customer = make_valid_customer()
    result = predict_churn(customer)

    assert result["churn_prediction"] == 0
    assert result["churn_probability"] == pytest.approx(0.2)
    assert result["threshold"] == 0.5

    mock_model.predict_proba.assert_called_once()

    model_input = mock_model.predict_proba.call_args.args[0]

    assert list(model_input.columns) == FEATURES
    assert model_input.shape == (1, len(FEATURES))

    mock_load.assert_called_once()


@patch("src.predict_churn.THRESHOLD_PATH")
@patch("src.predict_churn.joblib.load")
def test_predict_churn_at_risk(mock_load, mock_threshold_path):
    mock_model = Mock()
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
    mock_load.return_value = mock_model

    mock_threshold_path.read_text.return_value = "0.5"

    customer = make_valid_customer()
    result = predict_churn(customer)

    assert result["churn_prediction"] == 1
    assert result["churn_probability"] == pytest.approx(0.8)
    assert result["threshold"] == 0.5

