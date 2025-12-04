# prediction.py
import joblib
from hdbscan.prediction import approximate_predict

def predict(data):
    model = joblib.load("hdbscan_model.sav")
    return approximate_predict(model, data)[0]
