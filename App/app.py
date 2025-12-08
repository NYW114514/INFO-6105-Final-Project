import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model, encoders, and industry averages 
model = joblib.load("../Model/xgboost_model.pkl")
label_encoders = joblib.load("../Model/label_encoders.pkl")
industry_avg = joblib.load("../Model/industry_averages.pkl")

st.set_page_config(page_title="Layoff Risk Predictor", layout="wide")

st.title("Company Layoff Risk Prediction App")

st.write("""
Upload a **CSV file containing the engineered features** required by the trained model.
The app will run predictions and display the results.
""")

MODEL_FEATURES = [
    'industry',
    'stage',
    'funds_raised',
    'region',
    'recency',
    'events_deviation',
    'recency_deviation',
    'layoff_events_category',
    'funds_raised_binned'
]

uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

# CSV rediction section
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(df, use_container_width=True)

    # Check required columns
    missing_cols = [col for col in MODEL_FEATURES if col not in df.columns]
    if missing_cols:
        st.error(f"The uploaded CSV is missing required columns: {missing_cols}")
        st.stop()

    # Optional company name column
    company_col = df["company"] if "company" in df.columns else ["N/A"] * len(df)

    # Model input columns
    df_model = df[MODEL_FEATURES].copy()

    # Encode categorical columns using saved encoders
    df_encoded = df_model.copy()
    for col in ['industry', 'stage', 'region', 'layoff_events_category', 'funds_raised_binned']:
        df_encoded[col] = label_encoders[col].transform(df_encoded[col])

    # Predict
    preds = model.predict(df_encoded)
    probs = model.predict_proba(df_encoded)[:, 1]

    # Build output table
    results = pd.DataFrame({
        "Company": company_col,
        "Industry": df["industry"],
        "High Risk?": ["YES" if p == 1 else "NO" for p in preds],
        "Confidence": probs
    })

    def color_risk(val):
        return "color: red;" if val == "YES" else "color: green;"

    st.subheader("Prediction Results")
    st.dataframe(
        results.style.applymap(color_risk, subset=["High Risk?"]),
        use_container_width=True
    )

# Single Company Prediction
st.divider()
st.subheader("Single Company Prediction")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        industry = st.selectbox("Industry", df['industry'].unique() if uploaded_file else ["Healthcare", "Finance", "Tech"])
        stage = st.selectbox("Stage", df['stage'].unique() if uploaded_file else ["Series A", "Series B", "Series C"])
        funds_raised = st.number_input("Funds Raised (Million $)", min_value=0.0, step=100.0)
        region = st.selectbox("Region", df['region'].unique() if uploaded_file else ["North America", "Europe", "Asia"])

    with col2:
        latest_layoff_year = st.number_input("Latest Layoff Year", min_value=2000, max_value=2025, value=2024)
        layoff_count = st.number_input("Total Layoff Events", min_value=0, step=1)
        layoff_events_category = st.selectbox("Layoff Events Category",
                                              df['layoff_events_category'].unique() if uploaded_file else ["once", "twice", "four_plus"])
        funds_raised_binned = st.selectbox("Funds Raised Binned",
                                           df['funds_raised_binned'].unique() if uploaded_file else ["small", "medium", "large"])

    col_btn1, col_btn2 = st.columns(2)
    submitted = col_btn1.form_submit_button("Predict", use_container_width=True)
    reset = col_btn2.form_submit_button("Reset", use_container_width=True)

    if submitted and uploaded_file:
        # Compute engineered features
        recency = 2025 - latest_layoff_year

        if industry in industry_avg.index:
            ind_avg_row = industry_avg.loc[industry]
        else:
            ind_avg_row = industry_avg.mean()

        events_deviation = layoff_count - ind_avg_row['events_deviation']
        recency_deviation = recency - ind_avg_row['recency_deviation']

        user_data = pd.DataFrame({
            'industry': [industry],
            'stage': [stage],
            'funds_raised': [funds_raised],
            'region': [region],
            'recency': [recency],
            'events_deviation': [events_deviation],
            'recency_deviation': [recency_deviation],
            'layoff_events_category': [layoff_events_category],
            'funds_raised_binned': [funds_raised_binned]
        })

        # Encode using saved encoders
        user_encoded = user_data.copy()
        for col in ['industry', 'stage', 'region', 'layoff_events_category', 'funds_raised_binned']:
            user_encoded[col] = label_encoders[col].transform(user_encoded[col])

        # Predict
        pred = model.predict(user_encoded)[0]
        prob = model.predict_proba(user_encoded)[0, 1]

        st.markdown("---")
        colA, colB, colC = st.columns(3)
        colA.metric("Risk Level", "HIGH" if pred == 1 else "LOW")
        colB.metric("Confidence", f"{prob:.2%}")
        colC.metric("Prediction", "YES" if pred == 1 else "NO")

    if reset:
        st.rerun()
