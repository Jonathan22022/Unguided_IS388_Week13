# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from hdbscan.prediction import approximate_predict

# TITLE
st.title("HDBSCAN Clustering – Streamlit App")

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_excel("data/OTP_Time_Series_Master.xlsx")

st.subheader("Dataset Preview")
st.dataframe(df.head())

clean_cols = df.columns.str.replace("\n", "", regex=False).str.replace(" ", "").str.lower()

def find_col(keyword):
    for i, col in enumerate(clean_cols):
        if keyword in col:
            return df.columns[i]
    st.error(f"Kolom '{keyword}' tidak ditemukan!")
    st.stop()

departures = find_col("ontimedepartures")
arrivals = find_col("ontimearrivals")
cancellations = find_col("cancellations")
sectors = find_col("sectorsflown")

features = [departures, arrivals, cancellations, sectors]

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=features)

# -------------------------
# LOAD SCALER + MODEL
# -------------------------
scaler = joblib.load("scaler.sav")
model = joblib.load("hdbscan_model.sav")

X_scaled = scaler.transform(df[features])

# -------------------------
# CLUSTERING RESULT
# -------------------------
labels = model.labels_
df["Cluster"] = labels

st.subheader("Cluster Distribution")
st.write(df["Cluster"].value_counts())

st.dataframe(df[["Cluster"] + features].head())

# -------------------------
# USER INPUT PREDICTION
# -------------------------
st.markdown("### Prediksi Cluster (User Input)")

col1, col2 = st.columns(2)
with col1:
    dep = st.slider("OnTime Departures (%)", 0.0, 100.0, 70.0)
    arr = st.slider("OnTime Arrivals (%)", 0.0, 100.0, 65.0)
with col2:
    canc = st.slider("Cancellations (%)", 0.0, 10.0, 1.0)
    sectors_val = st.slider("Sectors Flown", 0.0, 500.0, 120.0)

if st.button("Predict Cluster"):
    user_input = np.array([[dep, arr, canc, sectors_val]])
    user_scaled = scaler.transform(user_input)

    # HDBSCAN PREDICTION
    cluster_label, _ = approximate_predict(model, user_scaled)
    st.success(f"Hasil prediksi cluster: {cluster_label[0]}")

# -------------------------
# PCA VISUALIZATION
# -------------------------
st.subheader("HDBSCAN PCA Visualization")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=labels,
    s=50,
    alpha=0.7
)

# Plot user input
if 'user_scaled' in locals():
    user_pca = pca.transform(user_scaled)
    ax.scatter(user_pca[0, 0], user_pca[0, 1], s=200, marker="X")

ax.set_title("HDBSCAN Clustering (PCA 2D)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

st.pyplot(fig)

