import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and scaler
# Ensure turbine_model.pkl and scaler.pkl are in the same folder
try:
    model = joblib.load('turbine_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model files not found! Please run your training script first.")

# --- UI Configuration ---
st.set_page_config(page_title="Turbine Health Monitor", layout="wide")

st.title("🚢 Thermal Turbine Health Dashboard")
st.markdown("""
Monitor real-time turbine conditions and predict **Remaining Useful Life (RUL)** and **Failure Probability** using Machine Learning.
""")

st.write("---")

# --- Sidebar Inputs ---
st.sidebar.header("🛠️ Sensor Live Data")
usage = st.sidebar.number_input("Total Usage (Hours)", min_value=0, max_value=50000, value=1200)
temp = st.sidebar.slider("Inlet Temperature (°C)", 100.0, 300.0, 160.0)
vib = st.sidebar.slider("Vibration Velocity (mm/s)", 0.0, 15.0, 2.1)
rpm = st.sidebar.number_input("Load (RPM)", min_value=0, max_value=5000, value=3000)

# --- Prediction Logic ---
# 1. Prepare and Scale Input
input_features = np.array([[usage, temp, vib, rpm]])
input_scaled = scaler.transform(input_features)

# 2. Generate Prediction
prediction = model.predict(input_scaled)

# Extract values and FORCE cast to standard Python float
# This solves the StreamlitAPIException: Progress Value has invalid type
risk_val = float(prediction[0][0])
rul_val = float(prediction[0][1])

# --- Display Results ---
col1, col2, col3 = st.columns(3)

with col1:
    risk_percentage = min(100.0, max(0.0, risk_val * 100))
    st.metric("Failure Risk", f"{risk_percentage:.2f}%")
    
    if risk_val > 0.75:
        st.error("🚨 CRITICAL: Immediate Shutdown Recommended")
    elif risk_val > 0.40:
        st.warning("⚠️ WARNING: Schedule Maintenance")
    else:
        st.success("✅ SAFE: Optimal Condition")

with col2:
    # Ensure RUL doesn't show negative numbers
    display_rul = max(0, int(rul_val))
    st.metric("Estimated RUL", f"{display_rul} Hours")

with col3:
    st.metric("Current Load", f"{rpm} RPM")

# --- Health Progress Bar ---
st.write("---")
st.subheader("Machine Health Index")

# Ensure the progress value is between 0.0 and 1.0
health_index = float(np.clip(1.0 - risk_val, 0.0, 1.0))

# Visual bar
st.progress(health_index)
st.caption(f"Current Health Integrity: {health_index*100:.1f}%")

# --- Data Table Summary ---
with st.expander("View Raw Input Feature Vector"):
    df_display = pd.DataFrame(input_features, columns=['Usage', 'Temp', 'Vib', 'RPM'])
    st.table(df_display)