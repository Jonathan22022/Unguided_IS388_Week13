import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

df = pd.read_excel("OTP_Time_Series_Master.xlsx")

features = [
    "OnTime Departures \n(%)",
    "OnTime Arrivals \n(%)",
    "Cancellations \n\n(%)",
    "Sectors Flown"
]

df = df.dropna(subset=features)

X = df[features]

# Fit scaler ulang
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Load model DBSCAN
dbscan_model = joblib.load("dbscan_model.sav")


# ---------------------------------------------
# STREAMLIT UI
# ---------------------------------------------

st.title("DBSCAN Clustering Prediction")
st.markdown("Masukkan nilai fitur untuk memprediksi cluster menggunakan model DBSCAN.")

col1, col2 = st.columns(2)

with col1:
    st.text("On-Time Performance")
    dep = st.slider("OnTime Departures (%)", 0.0, 100.0, 80.0)
    arr = st.slider("OnTime Arrivals (%)", 0.0, 100.0, 75.0)

with col2:
    st.text("Operational Data")
    canc = st.slider("Cancellations (%)", 0.0, 10.0, 1.0)
    sectors = st.slider("Sectors Flown", 0.0, 500.0, 120.0)

st.text("")

if st.button("Predict Cluster"):
    user_input = np.array([[dep, arr, canc, sectors]])
    user_scaled = scaler.transform(user_input)

    result = dbscan_model.fit_predict(user_scaled)
    st.text(f"Cluster result: {result[0]}")