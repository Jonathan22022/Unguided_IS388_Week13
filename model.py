import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

df = pd.read_excel('data/OTP_Time_Series_Master.xlsx')
df.head()

df.shape

df.info()

df.describe()

df.isna().sum()

df = df.replace("na", np.nan)
df = df.dropna() 

features = ["OnTime Departures \n(%)", "OnTime Arrivals \n(%)", "Cancellations \n\n(%)", "Sectors Flown"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

db = DBSCAN(eps=0.6, min_samples=5)
labels = db.fit_predict(X_scaled)

df["Cluster_DBSCAN"] = labels

df["Cluster_DBSCAN"].value_counts()

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, s=50)
plt.title("DBSCAN Clustering (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

import pickle

with open('modelDBSCAN.pkl', 'wb') as file:
    pickle.dump(db, file)

import joblib

joblib.dump(db, "dbscan_model.sav")