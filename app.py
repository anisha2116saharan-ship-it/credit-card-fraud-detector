import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# 1. Page Configuration
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("""
This system evaluates credit card transactions using a **Random Forest Classifier** trained on highly imbalanced financial transaction signals.
""")

# 2. Train Model in Memory (Cached so it loads instantly)
@st.cache_resource
def get_trained_model():
    # Simulates financial PCA features (V1-V20) with 0.5% fraud rate
    X, y = make_classification(
        n_samples=20000,
        n_features=20,
        n_informative=14,
        weights=[0.995, 0.005],
        random_state=42
    )
    clf = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42)
    clf.fit(X, y)
    return clf

model = get_trained_model()

# 3. Sidebar Controls for Inputs (Configured for INR ₹)
st.sidebar.header("🔍 Input Transaction Parameters")
amount = st.sidebar.number_input(
    "Transaction Amount (₹)", 
    min_value=10.0, 
    max_value=500000.0, 
    value=5000.0, 
    step=500.0
)

st.sidebar.subheader("PCA Anonymized Features")
v1 = st.sidebar.slider("Signal V1", -5.0, 5.0, 0.2)
v2 = st.sidebar.slider("Signal V2", -5.0, 5.0, -1.1)
v3 = st.sidebar.slider("Signal V3", -5.0, 5.0, 2.0)
v4 = st.sidebar.slider("Signal V4", -5.0, 5.0, -0.5)

# Assemble feature vector
transaction_features = np.zeros((1, 20))
transaction_features[0, 0] = v1
transaction_features[0, 1] = v2
transaction_features[0, 2] = v3
transaction_features[0, 3] = v4

# 4. Main Display & Evaluation
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Transaction Summary")
    summary_df = pd.DataFrame({
        "Parameter": ["Amount", "Signal V1", "Signal V2", "Signal V3", "Signal V4"],
        "Value": [f"₹{amount:,.2f}", v1, v2, v3, v4]
    })
    st.table(summary_df)

with col2:
    st.subheader("Risk Assessment")
    if st.button("Run Fraud Analysis", type="primary"):
        prediction = model.predict(transaction_features)[0]
        prob = model.predict_proba(transaction_features)[0][1] * 100

        st.metric(label="Calculated Fraud Probability", value=f"{prob:.2f}%")

        if prediction == 1 or prob >= 50.0:
            st.error("🚨 **ALERT: High Risk of Fraudulent Transaction Detected!**")
            st.write("Recommendation: Block transaction and trigger multi-factor authentication (OTP/MFA).")
        else:
            st.success("✅ **APPROVED: Transaction appears legitimate.**")
            st.write("Recommendation: Process transaction normally.")
