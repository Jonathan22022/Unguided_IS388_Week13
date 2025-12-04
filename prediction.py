# prediction.py
import joblib
from hdbscan import approximation as approx

def predict(data):
    model = joblib.load("hdbscan_model.sav")
    return approx.approximate_predict(model, data)[0]
