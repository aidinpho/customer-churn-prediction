import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from predict_churn import predict_churn, FEATURES


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📊 Customer Churn Prediction")

st.markdown(
    """
    Estimate a telecom customer's churn risk using a
    trained **Random Forest** machine learning model.

    Enter the customer's information below and select
    **Predict Churn Risk** to see the model's result.
    """
)

st.info(
    "This tool provides an estimate based on historical "
    "data. Predictions are not guarantees of future "
    "customer behavior."
)


# --------------------------------------------------
# Customer information form
# --------------------------------------------------

st.header("Customer Information")

st.caption(
    "Complete the sections below using the customer's "
    "available records."
)

with st.form("customer_form"):

    # ----------------------------------------------
    # Section 1: Customer profile
    # ----------------------------------------------

    st.subheader("👤 Customer Profile")

    profile_col1, profile_col2 = st.columns(2)

    with profile_col1:
        age = st.number_input(
            "Customer Age",
            min_value=15,
            max_value=55,
            value=30,
            step=1,
            help=(
                "Customer's age in years. "
                "The accepted range is 15–55."
            ),
        )

        age_group = st.selectbox(
            "Age Group",
            options=[1, 2, 3, 4, 5],
            index=2,
            format_func=lambda x: f"Group {x}",
            help=(
                "Select the customer's age category. "
                "Group 1 represents the younger age category, "
                "and Group 5 represents the older age category. "
                "Exact age ranges for each group are not available "
                "in the official dataset documentation."
            ),
        )


    with profile_col2:
        subscription_length = st.number_input(
            "Subscription Length",
            min_value=0,
            value=30,
            step=1,
            help=(
                "Length of the customer's subscription "
                "in months."
            ),
        )

        tariff_plan = st.selectbox(
            "Tariff Plan",
            options=[1, 2],
            index=0,
            format_func=lambda x: (
                "Plan 1" if x == 1 else "Plan 2"
            ),
            help=(
                "Select the tariff plan recorded for the customer. "
                "The exact differences between plans have not "
                "been verified."
            ),
        )
    st.divider()

    # ----------------------------------------------
    # Section 2: Calling behavior
    # ----------------------------------------------

    st.subheader("📞 Calling Behavior")

    call_col1, call_col2 = st.columns(2)

    with call_col1:
        call_failure = st.number_input(
            "Call Failures",
            min_value=0,
            value=8,
            step=1,
            help=(
                "Number of failed calls recorded "
                "for the customer."
            ),
        )

        seconds_of_use = st.number_input(
            "Total Seconds of Use",
            min_value=0,
            value=2500,
            step=1,
            help=(
                "Total recorded call usage in seconds."
            ),
        )

    with call_col2:
        frequency_of_use = st.number_input(
            "Frequency of Use",
            min_value=0,
            value=40,
            step=1,
            help=(
                "Number of calls recorded for "
                "the customer."
            ),
        )

        distinct_called_numbers = st.number_input(
            "Distinct Called Numbers",
            min_value=0,
            value=15,
            step=1,
            help=(
                "Number of distinct phone numbers "
                "the customer called."
            ),
        )

    st.divider()

    # ----------------------------------------------
    # Section 3: SMS and complaints
    # ----------------------------------------------

    st.subheader("💬 SMS & Customer Complaints")

    communication_col1, communication_col2 = (
        st.columns(2)
    )

    with communication_col1:
        frequency_of_sms = st.number_input(
            "Frequency of SMS",
            min_value=0,
            value=20,
            step=1,
            help=(
                "Number of SMS messages recorded "
                "for the customer."
            ),
        )

    with communication_col2:
        complains = st.selectbox(
            "Has the Customer Complained?",
            options=[0, 1],
            format_func=lambda value: (
                "No" if value == 0 else "Yes"
            ),
            help=(
                "Select whether a customer complaint "
                "is recorded: No = 0, Yes = 1."
            ),
        )

    st.divider()

    # ----------------------------------------------
    # Section 4: Billing and customer value
    # ----------------------------------------------

    st.subheader("💳 Billing & Customer Value")

    financial_col1, financial_col2 = st.columns(2)

    with financial_col1:
        charge_amount = st.number_input(
            "Charge Amount Level",
            min_value=0,
            max_value=10,
            value=2,
            step=1,
            help=(
                "An ordinal level from 0 (lowest) "
                "to 10 (highest) in this dataset. "
                "It is not a currency amount."
            ),
        )

    with financial_col2:
        customer_value = st.number_input(
            "Customer Value",
            min_value=0.0,
            value=200.0,
            step=1.0,
            help=(
                "Customer Value as recorded in the "
                "dataset. Its exact business unit and "
                "measurement timing are not established "
                "by this interface."
            ),
        )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Predict Churn Risk",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

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

    # Ensure that the expected model features are present.
    if set(customer_data) != set(FEATURES):
        st.error(
            "The input fields do not match the model's "
            "expected features. Please check the application."
        )

    else:
        try:
            with st.spinner(
                "Estimating customer churn risk..."
            ):
                result = predict_churn(customer_data)

            probability = result["churn_probability"]
            prediction = result["churn_prediction"]
            threshold = result["threshold"]

            # --------------------------------------
            # Results section
            # --------------------------------------

            st.divider()
            st.header("📈 Prediction Results")

            metric_col1, metric_col2 = st.columns(2)

            with metric_col1:
                st.metric(
                    "Estimated Churn Probability",
                    f"{probability:.1%}",
                )

            with metric_col2:
                st.metric(
                    "Decision Threshold",
                    f"{threshold:.1%}",
                )

            st.progress(
                max(0.0, min(1.0, probability)),
                text="Estimated churn probability",
            )

            st.markdown("### Model Assessment")

            if prediction == 1:
                st.error(
                    "**⚠️ Higher Risk**\n\n"
                    "The model flags this customer as "
                    "being at risk of churn."
                )
            else:
                st.success(
                    "**✅ Not Flagged as High Risk**\n\n"
                    "The estimated probability is below "
                    "the model's decision threshold."
                )

            st.caption(
                "The classification depends on the selected "
                "decision threshold. The probability is a "
                "model output and has not been established "
                "as a calibrated real-world probability."
            )

            with st.expander(
                "How should I interpret this result?"
            ):
                st.markdown(
                    f"""
                    - **Estimated probability:** {probability:.1%}
                    - **Decision threshold:** {threshold:.1%}
                    - **Model output:** {"Churn risk flagged" if prediction == 1 else "Churn risk not flagged"}

                    The model flags a customer when the
                    estimated probability is greater than
                    or equal to the decision threshold.

                    This result should be considered alongside
                    other customer information and should not
                    be treated as a guarantee of future behavior.
                    """
                )

        except (ValueError, TypeError) as error:
            st.error(
                f"Please check the customer information: {error}"
            )

        except Exception:
            st.error(
                "The prediction could not be completed. "
                "Please check the model files and application logs."
            )

