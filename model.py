import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ------------------------
# 1. TITLE
# ------------------------
st.title("DBSCAN Clustering Visualization App")

# ------------------------
# 2. LOAD DATA
# ------------------------
df = pd.read_excel("data/OTP_Time_Series_Master.xlsx")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Cleaning
df = df.replace("na", np.nan)
df = df.dropna()

# Feature columns
features = [
    "OnTime Departures \n(%)",
    "OnTime Arrivals \n(%)",
    "Cancellations \n\n(%)",
    "Sectors Flown"
]

X = df[features]

# ------------------------
# 3. LOAD MODEL
# ------------------------
model = joblib.load("dbscan_model.sav")

# Scaling (harus sama seperti training!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run clustering
labels = model.fit_predict(X_scaled)
df["Cluster_DBSCAN"] = labels

# ------------------------
# 4. SHOW CLUSTER RESULT
# ------------------------
st.subheader("Cluster Result")
st.write(df["Cluster_DBSCAN"].value_counts())

st.dataframe(df[["Cluster_DBSCAN"] + features].head())

# ------------------------
# 5. PCA VISUALIZATION
# ------------------------
st.subheader("DBSCAN Clustering Visualization (PCA 2D)")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=50)
ax.set_title("DBSCAN Clustering (PCA 2D)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

# tampilkan warna cluster
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

# IMPORTANT → tampilkan ke Streamlit
st.pyplot(fig)
