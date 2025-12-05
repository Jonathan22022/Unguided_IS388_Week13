import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def load_and_prepare():
    df = pd.read_excel("data/OTP_Time_Series_Master.xlsx")

    # Bersihkan nama kolom
    df.columns = (
        df.columns
        .str.replace("\n", " ", regex=False)
        .str.replace("  ", " ", regex=False)
        .str.strip()
    )

    # Kolom numerik
    num_cols = [
        'Sectors Scheduled',
        'Sectors Flown',
        'Cancellations',
        'Departures On Time',
        'Arrivals On Time',
        'Departures Delayed',
        'Arrivals Delayed',
        'OnTime Departures (%)',
        'OnTime Arrivals (%)',
        'Cancellations  (%)'
    ]

    df = df.replace("na", np.nan)
    df = df.dropna(subset=num_cols)

    data = df[num_cols]

    # Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    return df, scaled_data


def cluster_and_pca(scaled_data, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)

    # PCA 2D
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    return clusters, pca_result, kmeans, pca
