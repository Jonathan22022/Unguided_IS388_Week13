import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import joblib

df = pd.read_excel('data/OTP_Time_Series_Master.xlsx')

df = df.replace("na", np.nan)
df = df.dropna()

features = [
    "OnTime Departures \n(%)",
    "OnTime Arrivals \n(%)",
    "Cancellations \n\n(%)",
    "Sectors Flown"
]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

db = DBSCAN(eps=0.6, min_samples=5)
labels = db.fit_predict(X_scaled)

df["Cluster_DBSCAN"] = labels

print("Cluster Results:")
print(df["Cluster_DBSCAN"].value_counts())

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10,7))
plt.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=labels,
    cmap="viridis",
    s=60,
    alpha=0.8
)
plt.title("DBSCAN Clustering Visualization (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Cluster ID")
plt.show()

plt.figure(figsize=(6,4))
df["Cluster_DBSCAN"].value_counts().sort_index().plot(kind="bar", color="skyblue")
plt.title("Jumlah Data per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Jumlah Data")
plt.show()

df.groupby("Cluster_DBSCAN")[features].mean().plot(kind="bar", figsize=(12,6))
plt.title("Rata-rata Nilai Fitur per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Mean Value")
plt.xticks(rotation=0)
plt.show()

joblib.dump(db, "dbscan_model.sav")
joblib.dump(scaler, "scaler_dbscan.sav")

print("Model DBSCAN dan Scaler berhasil disimpan!")
