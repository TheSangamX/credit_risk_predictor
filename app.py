import os
import joblib
import pandas as pd
import streamlit as st
import altair as alt


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide"
)


# =========================================================
# MODEL METRICS
# =========================================================

METRICS = {
    "Accuracy": "85.17%",
    "Precision": "24.98%",
    "Recall": "41.79%",
    "F1 Score": "31.27%",
    "ROC-AUC": "75.99%"
}


# =========================================================
# TOP 20 FEATURE IMPORTANCE
# Obtained from final model analysis
# =========================================================

TOP_FEATURES = {
    "num__EXT_SOURCE_3": 0.062478,
    "num__EXT_SOURCE_2": 0.048455,
    "num__AMT_GOODS_PRICE": 0.019804,
    "num__EXT_SOURCE_1": 0.019637,
    "num__AMT_CREDIT": 0.012610,
    "num__DAYS_BIRTH": 0.006826,
    "num__AMT_ANNUITY": 0.004238,
    "num__DAYS_EMPLOYED": 0.003826,
    "cat__CODE_GENDER_F": 0.002707,
    "cat__NAME_EDUCATION_TYPE_Higher education": 0.002321,
    "cat__FLAG_OWN_CAR_N": 0.002181,
    "num__DAYS_ID_PUBLISH": 0.001773,
    "cat__NAME_FAMILY_STATUS_Married": 0.001443,
    "cat__NAME_CONTRACT_TYPE_Cash loans": 0.001372,
    "num__DAYS_LAST_PHONE_CHANGE": 0.001181,
    "num__AMT_REQ_CREDIT_BUREAU_QRT": 0.000878,
    "cat__CODE_GENDER_M": 0.000840,
    "num__FLAG_DOCUMENT_3": 0.000743,
    "cat__NAME_INCOME_TYPE_Working": 0.000685,
    "num__DAYS_REGISTRATION": 0.000610
}


# =========================================================
# HUMAN-READABLE FEATURE NAMES
# =========================================================

FEATURE_NAME_MAP = {

    "num__EXT_SOURCE_3":
        "External Source 3",

    "num__EXT_SOURCE_2":
        "External Source 2",

    "num__AMT_GOODS_PRICE":
        "Goods Price",

    "num__EXT_SOURCE_1":
        "External Source 1",

    "num__AMT_CREDIT":
        "Credit Amount",

    "num__DAYS_BIRTH":
        "Age",

    "num__AMT_ANNUITY":
        "Loan Annuity",

    "num__DAYS_EMPLOYED":
        "Days Employed",

    "cat__CODE_GENDER_F":
        "Gender: Female",

    "cat__NAME_EDUCATION_TYPE_Higher education":
        "Education: Higher Education",

    "cat__FLAG_OWN_CAR_N":
        "Owns Car: No",

    "num__DAYS_ID_PUBLISH":
        "Days Since ID Published",

    "cat__NAME_FAMILY_STATUS_Married":
        "Family Status: Married",

    "cat__NAME_CONTRACT_TYPE_Cash loans":
        "Contract Type: Cash Loan",

    "num__DAYS_LAST_PHONE_CHANGE":
        "Days Since Last Phone Change",

    "num__AMT_REQ_CREDIT_BUREAU_QRT":
        "Credit Bureau Requests: Quarterly",

    "cat__CODE_GENDER_M":
        "Gender: Male",

    "num__FLAG_DOCUMENT_3":
        "Document 3 Flag",

    "cat__NAME_INCOME_TYPE_Working":
        "Income Type: Working",

    "num__DAYS_REGISTRATION":
        "Days Since Registration"
}


# =========================================================
# CREATE FEATURE IMPORTANCE DATAFRAME
# =========================================================

feature_importance_df = (
    pd.DataFrame(
        list(TOP_FEATURES.items()),
        columns=["Feature", "Importance"]
    )
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(drop=True)
)


# =========================================================
# APPLY READABLE NAMES
# =========================================================

feature_importance_df["Feature"] = (
    feature_importance_df["Feature"]
    .map(FEATURE_NAME_MAP)
)


# =========================================================
# ADD RANK
# =========================================================

feature_importance_df.insert(
    0,
    "Rank",
    range(1, len(feature_importance_df) + 1)
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "credit_risk_model.joblib"
)


DEFAULTS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "app_defaults.joblib"
)


# =========================================================
# LOAD MODEL AND DEFAULTS
# =========================================================

@st.cache_resource
def load_resources():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    if not os.path.exists(DEFAULTS_PATH):
        raise FileNotFoundError(
            f"Defaults file not found:\n{DEFAULTS_PATH}"
        )

    saved_model = joblib.load(MODEL_PATH)

    defaults = joblib.load(DEFAULTS_PATH)

    # Convert Series to dictionary
    if isinstance(defaults, pd.Series):
        defaults = defaults.to_dict()

    # Convert DataFrame to dictionary
    elif isinstance(defaults, pd.DataFrame):
        defaults = defaults.iloc[0].to_dict()

    return (
        saved_model["pipeline"],
        saved_model["threshold"],
        defaults
    )


# =========================================================
# LOAD RESOURCES
# =========================================================

try:

    with st.spinner(
        "Loading machine learning model..."
    ):

        pipeline, threshold, defaults = load_resources()

except Exception as e:

    st.error(
        f"Error loading model resources: {e}"
    )

    st.stop()


# =========================================================
# HEADER
# =========================================================

st.title("💳 Credit Risk Predictor")

st.markdown(
    """
    ### Predict the probability of loan default

    Enter the applicant's information below and the trained
    Machine Learning model will estimate the credit risk.
    """
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📊 Model Information")

st.sidebar.write(
    "**Model:** HistGradientBoosting"
)

st.sidebar.write(
    "**ROC-AUC:** 0.760"
)

st.sidebar.write(
    f"**Decision Threshold:** {threshold:.2f}"
)

st.sidebar.divider()

st.sidebar.info(
    "This application uses the trained ML pipeline "
    "saved from the credit risk analysis."
)


# =========================================================
# INPUT SECTION
# =========================================================

col1, col2, col3 = st.columns(3)


# =========================================================
# PERSONAL INFORMATION
# =========================================================

with col1:

    st.markdown(
        "### 👤 Personal<br>Information",
        unsafe_allow_html=True
    )

    gender = st.selectbox(
        "Gender",
        ["F", "M"]
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    family_status = st.selectbox(
        "Family Status",
        [
            "Married",
            "Single / not married",
            "Civil marriage",
            "Separated",
            "Widow"
        ]
    )

    education = st.selectbox(
        "Education",
        [
            "Secondary / secondary special",
            "Higher education",
            "Incomplete higher",
            "Lower secondary",
            "Academic degree"
        ]
    )


# =========================================================
# FINANCIAL INFORMATION
# =========================================================

with col2:

    st.markdown(
        "### 💰 Financial<br>Information",
        unsafe_allow_html=True
    )

    income = st.number_input(
        "Annual Income",
        min_value=10000.0,
        value=180000.0,
        step=10000.0
    )

    credit_amount = st.number_input(
        "Credit Amount",
        min_value=1000.0,
        value=500000.0,
        step=10000.0
    )

    annuity = st.number_input(
        "Loan Annuity",
        min_value=1000.0,
        value=25000.0,
        step=1000.0
    )

    goods_price = st.number_input(
        "Goods Price",
        min_value=1000.0,
        value=500000.0,
        step=10000.0
    )


# =========================================================
# APPLICANT PROFILE
# =========================================================

with col3:

    st.markdown(
        "### 🏠 Applicant<br>Profile",
        unsafe_allow_html=True
    )

    own_car = st.selectbox(
        "Owns Car?",
        ["Y", "N"]
    )

    own_realty = st.selectbox(
        "Owns Property?",
        ["Y", "N"]
    )

    income_type = st.selectbox(
        "Income Type",
        [
            "Working",
            "Commercial associate",
            "Pensioner",
            "State servant",
            "Unemployed",
            "Student",
            "Businessman",
            "Maternity leave"
        ]
    )

    housing_type = st.selectbox(
        "Housing Type",
        [
            "House / apartment",
            "With parents",
            "Municipal apartment",
            "Rented apartment",
            "Office apartment",
            "Co-op apartment"
        ]
    )


st.divider()


# =========================================================
# PREDICTION
# =========================================================

if st.button(
    "🔍 Predict Credit Risk",
    use_container_width=True
):

    # -----------------------------------------------------
    # BASIC INPUT VALIDATION
    # -----------------------------------------------------

    if annuity > income:

        st.warning(
            "⚠️ Loan annuity should not exceed annual income."
        )

        st.stop()

    if goods_price > credit_amount:

        st.warning(
            "⚠️ Goods price should not exceed the credit amount."
        )

        st.stop()


    # -----------------------------------------------------
    # CREATE INPUT FROM SAVED DEFAULTS
    # -----------------------------------------------------

    input_data = defaults.copy()


    # -----------------------------------------------------
    # OVERRIDE USER INPUTS
    # -----------------------------------------------------

    input_data["CODE_GENDER"] = gender

    input_data["DAYS_BIRTH"] = -(age * 365)

    input_data["NAME_FAMILY_STATUS"] = family_status

    input_data["NAME_EDUCATION_TYPE"] = education

    input_data["AMT_INCOME_TOTAL"] = income

    input_data["AMT_CREDIT"] = credit_amount

    input_data["AMT_ANNUITY"] = annuity

    input_data["AMT_GOODS_PRICE"] = goods_price

    input_data["FLAG_OWN_CAR"] = own_car

    input_data["FLAG_OWN_REALTY"] = own_realty

    input_data["NAME_INCOME_TYPE"] = income_type

    input_data["NAME_HOUSING_TYPE"] = housing_type


    # -----------------------------------------------------
    # CREATE DATAFRAME
    # -----------------------------------------------------

    input_df = pd.DataFrame(
        [input_data]
    )


    # -----------------------------------------------------
    # ALIGN FEATURES WITH TRAINED PIPELINE
    # -----------------------------------------------------

    if hasattr(
        pipeline,
        "feature_names_in_"
    ):

        missing_features = set(
            pipeline.feature_names_in_
        ) - set(
            input_df.columns
        )

        if missing_features:

            st.error(
                "Missing model features: "
                + ", ".join(
                    sorted(missing_features)
                )
            )

            st.stop()

        input_df = input_df[
            pipeline.feature_names_in_
        ]


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    try:

        probability = pipeline.predict_proba(
            input_df
        )[0, 1]


        prediction = int(
            probability >= threshold
        )


        probability_percent = (
            probability * 100
        )


        # =================================================
        # PREDICTION RESULT
        # =================================================

        st.subheader(
            "🎯 Prediction Result"
        )


        result_col1, result_col2 = st.columns(2)


        # -------------------------------------------------
        # PROBABILITY
        # -------------------------------------------------

        with result_col1:

            st.metric(
                "Predicted Default Probability",
                f"{probability_percent:.2f}%"
            )


        # -------------------------------------------------
        # RISK CLASSIFICATION
        # -------------------------------------------------

        with result_col2:

            if prediction == 1:

                st.error(
                    "⚠️ HIGH CREDIT RISK"
                )

                st.write(
                    "The applicant has been classified "
                    "as having a higher risk of default."
                )

                st.write(
                    f"Predicted probability is at or above "
                    f"the {threshold * 100:.0f}% "
                    f"decision threshold."
                )

            else:

                st.success(
                    "✅ LOWER CREDIT RISK"
                )

                st.write(
                    "The applicant has been classified "
                    "as having a lower risk of default."
                )

                st.write(
                    f"Predicted probability is below "
                    f"the {threshold * 100:.0f}% "
                    f"decision threshold."
                )


        # -------------------------------------------------
        # PROBABILITY BAR
        # -------------------------------------------------

        st.progress(
            min(
                float(probability),
                1.0
            )
        )


        st.caption(
            f"Decision threshold: "
            f"{threshold:.2f} "
            f"({threshold * 100:.0f}%)"
        )


        # =================================================
        # MODEL PERFORMANCE
        # =================================================

        st.divider()

        st.subheader(
            "📈 Model Performance"
        )


        metric_columns = st.columns(
            len(METRICS)
        )


        for column, (
            metric_name,
            metric_value
        ) in zip(
            metric_columns,
            METRICS.items()
        ):

            column.metric(
                metric_name,
                metric_value
            )


        # =================================================
        # TOP 20 FEATURE IMPORTANCE
        # =================================================

        st.divider()

        st.subheader(
            "🔎 Top 20 Most Important Features"
        )


        st.caption(
            "Features identified as most influential "
            "during the final model analysis. "
            "Higher importance indicates greater "
            "contribution to the model's predictions."
        )


        # -------------------------------------------------
        # BAR CHART (STRICT DESCENDING ORDER)
        # -------------------------------------------------

        chart = (
            alt.Chart(feature_importance_df)
            .mark_bar()
            .encode(
                x=alt.X('Importance:Q', title='Importance'),
                y=alt.Y('Feature:N', sort=feature_importance_df['Feature'].tolist(), title='Feature'),
                tooltip=['Rank', 'Feature', 'Importance']
            )
            .properties(height=650)
        )

        st.altair_chart(chart, use_container_width=True)


        # -------------------------------------------------
        # FEATURE IMPORTANCE TABLE
        # -------------------------------------------------

        st.subheader(
            "📋 Feature Importance Details"
        )


        # Keep table in descending order
        # Highest importance = Rank 1

        display_df = (
            feature_importance_df.copy()
        )


        display_df["Importance"] = (
            display_df["Importance"]
            .map(
                lambda x: f"{x:.6f}"
            )
        )


        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


    # =====================================================
    # PREDICTION ERROR
    # =====================================================

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )