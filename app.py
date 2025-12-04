import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("DBSCAN Clustering Prediction + Visualization")

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

# Cari nama kolom
departures = find_col("ontimedepartures")
arrivals = find_col("ontimearrivals")
cancellations = find_col("cancellations")
sectors = find_col("sectorsflown")

features = [departures, arrivals, cancellations, sectors]

# Convert ke numeric
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=features)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Load model
dbscan_model = joblib.load("dbscan_model.sav")

# Predict cluster untuk dataset asli
labels = dbscan_model.fit_predict(X_scaled)
df["Cluster"] = labels

st.subheader("Cluster Result")
st.write(df["Cluster"].value_counts())

# ---------------------------
#  FORM INPUT USER
# ---------------------------
st.markdown("### Masukkan nilai fitur untuk memprediksi cluster:")

col1, col2 = st.columns(2)

with col1:
    dep = st.slider("OnTime Departures (%)", 0.0, 100.0, 80.0)
    arr = st.slider("OnTime Arrivals (%)", 0.0, 100.0, 75.0)

with col2:
    canc = st.slider("Cancellations (%)", 0.0, 10.0, 1.0)
    sectors_val = st.slider("Sectors Flown", 0.0, 500.0, 120.0)

user_result = None

if st.button("Predict Cluster"):
    user_input = np.array([[dep, arr, canc, sectors_val]])
    user_scaled = scaler.transform(user_input)
    user_result = dbscan_model.fit_predict(user_scaled)[0]
    st.success(f"Cluster result: {user_result}")

# ---------------------------
#  SCATTER VISUALIZATION PCA
# ---------------------------
st.subheader("DBSCAN Clustering Visualization (PCA 2D)")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=50, alpha=0.7)

# Tambahkan titik user (warna merah)
if user_result is not None:
    user_pca = pca.transform(user_scaled)
    ax.scatter(user_pca[0, 0], user_pca[0, 1], color="red", s=200, marker="X", label="User Input")

ax.set_title("DBSCAN Clustering (PCA 2D)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

# Legend cluster
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

if user_result is not None:
    ax.legend()

st.pyplot(fig)
