import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

st.title("DBSCAN Clustering Prediction")

# Load data
df = pd.read_excel("data/OTP_Time_Series_Master.xlsx")

# Bersihkan nama kolom
df = df.rename(columns={
    "OnTime Departures \n(%)": "Departures",
    "OnTime Arrivals \n(%)": "Arrivals",
    "Cancellations \n\n(%)": "Cancellations",
    "Sectors Flown": "Sectors"
})

features = ["Departures", "Arrivals", "Cancellations", "Sectors"]

# Hapus baris kosong
df = df.dropna(subset=features)

# Pastikan numerik
df[features] = df[features].astype(float)

# Fit scaler dari data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Load DBSCAN model
dbscan_model = joblib.load("dbscan_model.sav")

st.markdown("Masukkan nilai fitur untuk memprediksi cluster menggunakan model DBSCAN.")

col1, col2 = st.columns(2)

with col1:
    dep = st.slider("OnTime Departures (%)", 0.0, 100.0, 80.0)
    arr = st.slider("OnTime Arrivals (%)", 0.0, 100.0, 75.0)

with col2:
    canc = st.slider("Cancellations (%)", 0.0, 10.0, 1.0)
    sectors = st.slider("Sectors Flown", 0.0, 500.0, 120.0)

st.text("")

if st.button("Predict Cluster"):
    user_input = np.array([[dep, arr, canc, sectors]])
    user_scaled = scaler.transform(user_input)

    result = dbscan_model.fit_predict(user_scaled)

    st.success(f"Cluster result: {result[0]}")
