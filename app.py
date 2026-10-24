import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from predict_churn import predict_churn, FEATURES

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer information to estimate the risk "
    "of customer churn."
)

st.info(
    "This tool estimates churn risk using a trained "
    "Random Forest model. Predictions are estimates, "
    "not guarantees of future customer behavior."
)

st.subheader("Customer Information")

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        call_failure = st.number_input(
            "Call Failure",
            min_value=0,
            value=8,
            step=1,
        )
        complains = st.selectbox(
            "Complains",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )
        subscription_length = st.number_input(
            "Subscription Length",
            min_value=0,
            value=30,
            step=1,
        )
        charge_amount = st.number_input(
            "Charge Amount",
            min_value=0,
            value=2,
            step=1,
        )

    with col2:
        seconds_of_use = st.number_input(
            "Seconds of Use",
            min_value=0,
            value=2500,
            step=1,
        )
        frequency_of_use = st.number_input(
            "Frequency of use",
            min_value=0,
            value=40,
            step=1,
        )
        frequency_of_sms = st.number_input(
            "Frequency of SMS",
            min_value=0,
            value=20,
            step=1,
        )
        distinct_called_numbers = st.number_input(
            "Distinct Called Numbers",
            min_value=0,
            value=15,
            step=1,
        )

    with col3:
        age_group = st.selectbox(
            "Age Group",
            options=[1, 2, 3, 4, 5],
            index=2,
        )
        tariff_plan = st.selectbox(
            "Tariff Plan",
            options=[1, 2],
            index=0,
        )
        age = st.number_input(
            "Age",
            min_value=15,
            max_value=55,
            value=30,
            step=1,
        )
        customer_value = st.number_input(
            "Customer Value",
            min_value=0.0,
            value=200.0,
            step=1.0,
        )

    submitted = st.form_submit_button(
        "Predict Churn Risk",
        type="primary",
        use_container_width=True,
    )

if submitted:
    customer_data = {
        "Call Failure": call_failure,
        "Complains": complains,
        "Subscription Length": subscription_length,
        "Charge Amount": charge_amount,
        "Seconds of Use": seconds_of_use,
        "Frequency of use": frequency_of_use,
        "Frequency of SMS": frequency_of_sms,
        "Distinct Called Numbers": distinct_called_numbers,
        "Age Group": age_group,
        "Tariff Plan": tariff_plan,
        "Age": age,
        "Customer Value": customer_value,
    }

    try:
        result = predict_churn(customer_data)

        probability = result["churn_probability"]
        prediction = result["churn_prediction"]
        threshold = result["threshold"]

        st.divider()
        st.subheader("Prediction Results")

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.metric(
                "Estimated Churn Probability",
                f"{probability:.1%}",
            )

        with metric_col2:
            st.metric(
                "Decision Threshold",
                f"{threshold:.0%}",
            )

        st.progress(probability)

        if prediction == 1:
            st.error(
                "⚠️ High Risk: The model flags this customer "
                "as being at risk of churn."
            )
        else:
            st.success(
                "✅ Lower Risk: The model does not flag this "
                "customer as being at risk of churn."
            )

        st.caption(
            "The classification depends on the model's "
            "decision threshold. This is a prediction, "
            "not a certainty."
        )

    except (ValueError, TypeError) as error:
        st.error(f"Invalid customer information: {error}")

    except Exception as error:
        st.error(
            "The prediction could not be completed. "
            "Check the model files and application logs."
        )
        st.exception(error)