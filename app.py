import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

st.title("DBSCAN Clustering Prediction")

df = pd.read_excel("data/OTP_Time_Series_Master.xlsx")

st.write("Columns in Excel:", df.columns.tolist())

clean_cols = df.columns.str.replace("\n", "", regex=False).str.replace(" ", "").str.lower()

def find_col(keyword):
    matches = [df.columns[i] for i, col in enumerate(clean_cols) if keyword in col]
    if matches:
        return matches[0]
    else:
        st.error(f"Kolom dengan keyword '{keyword}' tidak ditemukan!")
        st.stop()

departures = find_col("ontimedepartures")
arrivals = find_col("ontimearrivals")
cancellations = find_col("cancellations")
sectors = find_col("sectorsflown")

features = [departures, arrivals, cancellations, sectors]

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=features)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

dbscan_model = joblib.load("dbscan_model.sav")

st.markdown("Masukkan nilai fitur untuk memprediksi cluster menggunakan model DBSCAN.")

col1, col2 = st.columns(2)

with col1:
    dep = st.slider("OnTime Departures (%)", 0.0, 100.0, 80.0)
    arr = st.slider("OnTime Arrivals (%)", 0.0, 100.0, 75.0)

with col2:
    canc = st.slider("Cancellations (%)", 0.0, 10.0, 1.0)
    sectors_val = st.slider("Sectors Flown", 0.0, 500.0, 120.0)

if st.button("Predict Cluster"):
    user_input = np.array([[dep, arr, canc, sectors_val]])
    user_scaled = scaler.transform(user_input)
    result = dbscan_model.fit_predict(user_scaled)
    st.success(f"Cluster result: {result[0]}")
