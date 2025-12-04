# model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import hdbscan

# -------------------------------
# LOAD DATASET
# -------------------------------
df = pd.read_excel("data/OTP_Time_Series_Master.xlsx")

# Clean column names
clean_cols = df.columns.str.replace("\n", "", regex=False).str.replace(" ", "").str.lower()

def find_col(keyword):
    for i, col in enumerate(clean_cols):
        if keyword in col:
            return df.columns[i]
    raise Exception(f"Kolom '{keyword}' tidak ditemukan!")

# Kolom yang dipakai
departures = find_col("ontimedepartures")
arrivals = find_col("ontimearrivals")
cancellations = find_col("cancellations")
sectors = find_col("sectorsflown")

features = [departures, arrivals, cancellations, sectors]

# Convert numeric
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=features)

# -------------------------------
# SCALING
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# -------------------------------
# TRAIN HDBSCAN MODEL
# -------------------------------
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=10,
    min_samples=3,
    cluster_selection_epsilon=0.5
)

clusterer.fit(X_scaled)

# -------------------------------
# SAVE MODEL & SCALER
# -------------------------------
joblib.dump(clusterer, "hdbscan_model.sav")
joblib.dump(scaler, "scaler.sav")

print("Model HDBSCAN dan scaler berhasil disimpan!")
