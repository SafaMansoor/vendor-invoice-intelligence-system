import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


RISK_FEATURES = (
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Invoice Intelligence System",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD MODELS (CACHED)
# ============================================================

@st.cache_resource
def load_models():
    base_dir = Path(__file__).resolve().parent / "models"
    freight_m = joblib.load(base_dir / "predict_freight_model.pkl")
    risk_m = joblib.load(base_dir / "invoice_risk_model.pkl")
    risk_s = joblib.load(base_dir / "risk_scaler.pkl")
    return freight_m, risk_m, risk_s

freight_model, risk_model, risk_scaler = load_models()

model_features = getattr(risk_model, "feature_names_in_", None)
scaler_features = getattr(risk_scaler, "feature_names_in_", None)
risk_model_ready = (
    model_features is not None
    and scaler_features is not None
    and tuple(model_features) == RISK_FEATURES
    and tuple(scaler_features) == RISK_FEATURES
)



# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 700;
    }

    .footer {
        text-align: center;
        padding: 20px;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🧾 Invoice Intelligence")

    st.caption(
        "Machine Learning based invoice analysis "
        "and business decision-support system."
    )

    st.divider()

    st.subheader("🤖 ML Models")

    st.write("🚚 Freight Prediction")
    st.caption("Linear Regression")

    st.write("🚨 Risk Detection")
    st.caption("Random Forest Classifier")

    st.divider()

    st.subheader("🛠️ Technology")

    st.write("🐍 Python")
    st.write("🐼 Pandas")
    st.write("📊 Scikit-learn")
    st.write("💾 Joblib")
    st.write("🎨 Streamlit")

    st.divider()

    st.subheader("📌 System Status")

    st.success("Freight Model Loaded")
    if risk_model_ready:
        st.success("Risk Model Ready")
    else:
        st.error(
            "Risk model and scaler do not match the five training features. "
            "Run the updated training notebook and reload the app."
        )
    if risk_model_ready:
        st.success("Application Ready")
    else:
        st.warning("Freight prediction is ready; risk prediction needs retraining.")

    st.divider()

    st.caption(
        "⚠️ Risk predictions are intended for "
        "manual-review prioritization and should "
        "not be treated as definitive fraud evidence."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    "🧾 Invoice Intelligence System"
    "</div>",
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    "AI-powered Freight Cost Prediction & "
    "Invoice Risk Detection"
    "</div>",
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# PROJECT OVERVIEW
# ============================================================

st.subheader("📊 Project Overview")

overview1, overview2, overview3, overview4 = st.columns(4)

with overview1:

    st.metric(
        "🚚 Workflow",
        "Freight Prediction"
    )

with overview2:

    st.metric(
        "🚨 Workflow",
        "Risk Detection"
    )

with overview3:

    st.metric(
        "🤖 ML Models",
        "2"
    )

with overview4:

    st.metric(
        "⚡ Framework",
        "Streamlit"
    )


# ============================================================
# ABOUT PROJECT
# ============================================================

with st.expander("📘 About This Project"):

    st.write(
        """
        Invoice Intelligence System is an end-to-end
        Machine Learning application designed to analyze
        invoice-related information and support business
        decision making.

        The application provides two ML workflows:

        1. Freight Cost Prediction

        Predicts expected freight cost from invoice amount
        using a trained Linear Regression model.

        2. Invoice Risk Detection

        Analyzes invoice characteristics and predicts
        whether an invoice may require manual review.

        The risk model uses a Random Forest Classifier
        and provides a model probability score.

        This system is a decision-support application.
        Final financial or operational decisions should
        involve human review.
        """
    )


# ============================================================
# HOW SYSTEM WORKS
# ============================================================

with st.expander("⚙️ How The System Works"):

    st.subheader("End-to-End ML Workflow")

    step1, step2, step3, step4, step5 = st.columns(5)

    with step1:

        st.info(
            "1️⃣\n\n"
            "Invoice Data"
        )

    with step2:

        st.info(
            "2️⃣\n\n"
            "Preprocessing"
        )

    with step3:

        st.info(
            "3️⃣\n\n"
            "ML Model"
        )

    with step4:

        st.info(
            "4️⃣\n\n"
            "Prediction"
        )

    with step5:

        st.info(
            "5️⃣\n\n"
            "Business Review"
        )


# ============================================================
# TABS
# ============================================================

tab1, tab2 = st.tabs(
    [
        "🚚 Freight Prediction",
        "🚨 Invoice Risk Detection"
    ]
)


# ============================================================
# TAB 1 - FREIGHT PREDICTION
# ============================================================

with tab1:

    st.markdown(
        '<div class="section-title">'
        "🚚 Freight Cost Prediction"
        "</div>",
        unsafe_allow_html=True
    )

    st.write(
        "Estimate the expected freight cost based on "
        "the invoice amount."
    )

    st.divider()

    input_col, model_col = st.columns(
        [2, 1]
    )

    with input_col:

        st.subheader("📥 Invoice Input")

        invoice_dollars = st.number_input(
            "💰 Invoice Amount ($)",
            min_value=0.0,
            value=1000.0,
            step=100.0,
            key="freight_invoice_amount",
            help="Enter the total invoice amount."
        )

    with model_col:

        st.subheader("🤖 Model Information")

        st.info(
            "Model: Linear Regression\n\n"
            "Task: Regression\n\n"
            "Output: Freight Cost"
        )

    st.divider()

    if st.button(
        "🔮 Predict Freight Cost",
        use_container_width=True,
        type="primary",
        key="freight_predict_button"
    ):

        if invoice_dollars <= 0:

            st.warning(
                "⚠️ Invoice Amount must be greater than $0."
            )

        else:

            input_data = pd.DataFrame(
                {
                    "Dollars": [
                        invoice_dollars
                    ]
                }
            )

            prediction = freight_model.predict(
                input_data
            )[0]

            prediction = max(
                0.0,
                float(prediction)
            )

            freight_ratio = (
                prediction /
                invoice_dollars
            ) * 100

            st.divider()

            st.subheader(
                "🎯 Prediction Results"
            )

            result1, result2, result3 = st.columns(3)

            with result1:

                st.metric(
                    "💰 Invoice Amount",
                    f"${invoice_dollars:,.2f}"
                )

            with result2:

                st.metric(
                    "🚚 Predicted Freight",
                    f"${prediction:,.2f}"
                )

            with result3:

                st.metric(
                    "📊 Freight Ratio",
                    f"{freight_ratio:.2f}%"
                )

            st.success(
                f"✅ Estimated Freight Cost: "
                f"${prediction:,.2f}"
            )

            st.info(
                f"For an invoice amount of "
                f"${invoice_dollars:,.2f}, the model "
                f"estimates approximately "
                f"${prediction:,.2f} in freight cost."
            )

            with st.expander(
                "📋 View Prediction Input"
            ):

                freight_summary = pd.DataFrame(
                    {
                        "Feature": [
                            "Invoice Amount"
                        ],
                        "Value": [
                            f"${invoice_dollars:,.2f}"
                        ]
                    }
                )

                st.dataframe(
                    freight_summary,
                    use_container_width=True,
                    hide_index=True
                )


# ============================================================
# TAB 2 - RISK DETECTION
# ============================================================
with tab2:
    st.markdown('<div class="section-title">🚨 Invoice Risk Detection</div>', unsafe_allow_html=True)
    st.write("Analyze invoice characteristics and identify invoices that may require manual review.")
    st.divider()
    
    st.subheader("📥 Invoice Information")
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        invoice_quantity = st.number_input(
            "📦 Invoice Quantity", min_value=0.0, value=100.0, step=1.0, 
            key="risk_invoice_quantity", help="Quantity associated with the invoice."
        )
        invoice_dollars = st.number_input(
            "💰 Invoice Dollars ($)", min_value=0.0, value=1000.0, step=100.0, 
            key="risk_invoice_dollars", help="Total invoice dollar amount."
        )
        Freight = st.number_input(
            "🚚 Freight ($)", min_value=0.0, value=50.0, step=10.0, 
            key="risk_freight", help="Freight amount associated with the invoice."
        )

    with risk_col2:
        total_item_quantity = st.number_input(
            "📦 Total Item Quantity", min_value=0.0, value=100.0, step=1.0, 
            key="risk_total_item_quantity", help="Total quantity of all items."
        )
        total_item_dollars = st.number_input(
            "💵 Total Item Dollars ($)", min_value=0.0, value=1000.0, step=100.0, 
            key="risk_total_item_dollars", help="Total monetary value of items."
        )

    st.divider()
    
    if st.button("Analyze Risk", type="primary", key="analyze_risk_btn"):
        if invoice_quantity <= 0 or invoice_dollars <= 0:
            st.warning("⚠️ Invoice Quantity and Invoice Dollars must be greater than 0.")
        elif not risk_model_ready:
            st.error(
                "Risk prediction is unavailable because the saved model and "
                "scaler do not match the five-feature training workflow. "
                "Run the updated training notebook, then restart the app."
            )
        else:
            risk_input = pd.DataFrame(
                [[
                    invoice_quantity,
                    invoice_dollars,
                    Freight,
                    total_item_quantity,
                    total_item_dollars,
                ]],
                columns=RISK_FEATURES,
            )
            
            # Scale input using the trained StandardScaler
            risk_input_scaled = pd.DataFrame(
                risk_scaler.transform(risk_input),
                columns=risk_input.columns
            )
            
            # Explicit decision threshold assignment with probability score
            risk_probability = float(risk_model.predict_proba(risk_input_scaled)[0][1])
            DECISION_THRESHOLD = 0.50
            risk_prediction = 1 if risk_probability >= DECISION_THRESHOLD else 0
            
            # Display results
            st.subheader("📊 Analysis Result")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("Risk Probability", f"{risk_probability * 100:.1f}%")
            with res_col2:
                st.metric("Decision Threshold", f"{DECISION_THRESHOLD * 100:.0f}%")
            st.progress(min(1.0, max(0.0, risk_probability)))
            
            if risk_prediction == 1:
                st.error("⚠️ **High Risk Flagged:** This invoice requires manual review!")
            else:
                st.success("✅ **Low Risk:** Invoice looks standard.")
# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        🧾 <b>Invoice Intelligence System</b><br>
        End-to-End Machine Learning Decision-Support Application
        <br><br>
        ⚠️ Risk predictions should be reviewed by a human
        before financial or operational decisions.
    </div>
    """,
    unsafe_allow_html=True
)