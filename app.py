# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("DBSCAN Clustering Prediction + Visualization")

# -------------------------------------
# Load dataset
# -------------------------------------
df = pd.read_excel("data/OTP_Time_Series_Master.xlsx")

clean_cols = df.columns.str.replace("\n", "", regex=False).str.replace(" ", "").str.lower()

def find_col(keyword):
    matches = [df.columns[i] for i, col in enumerate(clean_cols) if keyword in col]
    if matches:
        return matches[0]
    else:
        st.error(f"Kolom '{keyword}' tidak ditemukan!")
        st.stop()

departures = find_col("ontimedepartures")
arrivals = find_col("ontimearrivals")
cancellations = find_col("cancellations")
sectors = find_col("sectorsflown")

features = [departures, arrivals, cancellations, sectors]

# Convert numeric
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=features)

X = df[features]

# -------------------------------------
# Load Model & Scaler
# -------------------------------------
model = joblib.load("dbscan_model.sav")
scaler = joblib.load("scaler_dbscan.sav")

X_scaled = scaler.transform(X)
labels = model.fit_predict(X_scaled)
df["Cluster"] = labels

st.subheader("Cluster Summary")
st.write(df["Cluster"].value_counts())

st.dataframe(df[["Cluster"] + features].head())

# -------------------------------------
# User Input Form
# -------------------------------------
st.markdown("### Prediksi Cluster Berdasarkan Input User")

col1, col2 = st.columns(2)

with col1:
    dep = st.slider("OnTime Departures (%)", 0.0, 100.0, 80.0)
    arr = st.slider("OnTime Arrivals (%)", 0.0, 100.0, 75.0)

with col2:
    canc = st.slider("Cancellations (%)", 0.0, 10.0, 1.0)
    sec = st.slider("Sectors Flown", 0.0, 500.0, 150.0)

user_cluster = None

if st.button("Predict Cluster"):
    user_input = np.array([[dep, arr, canc, sec]])
    user_scaled = scaler.transform(user_input)
    user_cluster = model.fit_predict(user_scaled)[0]
    st.success(f"Cluster Prediction: **{user_cluster}**")

# -------------------------------------
# PCA Visualization
# -------------------------------------
st.subheader("Visualisasi DBSCAN dengan PCA")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=labels, s=60, alpha=0.7
)

if user_cluster is not None:
    user_pca = pca.transform(user_scaled)
    ax.scatter(
        user_pca[0, 0], user_pca[0, 1],
        s=200, marker="X", color="red",
        label="User Input"
    )

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("DBSCAN Clustering (PCA)")

legend = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend)

if user_cluster is not None:
    ax.legend()

st.pyplot(fig)
