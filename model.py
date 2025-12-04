# model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_excel("data/OTP_Time_Series_Master.xlsx")

# Cleaning column names
clean_cols = df.columns.str.replace("\n", "", regex=False).str.replace(" ", "").str.lower()

def find_col(keyword):
    matches = [df.columns[i] for i, col in enumerate(clean_cols) if keyword in col]
    if matches:
        return matches[0]
    else:
        raise ValueError(f"Kolom dengan keyword '{keyword}' tidak ditemukan!")

# Cari kolom fitur
departures = find_col("ontimedepartures")
arrivals = find_col("ontimearrivals")
cancellations = find_col("cancellations")
sectors = find_col("sectorsflown")

features = [departures, arrivals, cancellations, sectors]

# Convert ke numeric
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=features)

X = df[features]

# -----------------------------
# 2. Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 3. DBSCAN Model
# -----------------------------
model = DBSCAN(eps=0.8, min_samples=5)
model.fit(X_scaled)

# -----------------------------
# 4. Save model & scaler
# -----------------------------
joblib.dump(model, "dbscan_model.sav")
joblib.dump(scaler, "scaler_dbscan.sav")

print("Model & Scaler saved successfully!")
