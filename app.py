import streamlit as st
import matplotlib.pyplot as plt
from model import load_and_prepare, cluster_and_pca

st.set_page_config(page_title="KMeans PCA Clustering", layout="wide")

st.title("KMeans Clustering with PCA Visualization (2D)")
st.write("Upload dataset otomatis di-load dari file `OTP_Time_Series_Master.xlsx`.")

# Load dataset
df, scaled_data = load_and_prepare()

# Input K
k = st.slider("Pilih jumlah cluster (k)", min_value=2, max_value=10, value=3)

# Process clustering
clusters, pca_result, kmeans, pca = cluster_and_pca(scaled_data, k)

# Tambahkan ke dataframe
df["Cluster"] = clusters
df["PCA1"] = pca_result[:, 0]
df["PCA2"] = pca_result[:, 1]

st.subheader("🔍 Hasil PCA + Clustering")
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df["PCA1"], df["PCA2"], c=df["Cluster"])
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_title(f"PCA 2D Visualization with k={k}")
plt.colorbar(scatter, ax=ax)

st.pyplot(fig)

st.subheader("📄 Contoh Dataframe")
st.dataframe(df.head())
