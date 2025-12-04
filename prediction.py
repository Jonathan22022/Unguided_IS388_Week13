# prediction.py
import joblib

def predict(data):
    model = joblib.load("dbscan_model.sav")
    scaler = joblib.load("scaler_dbscan.sav")

    data_scaled = scaler.transform(data)
    prediction = model.fit_predict(data_scaled)

    return prediction
