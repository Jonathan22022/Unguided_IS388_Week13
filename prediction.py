import joblib

def predict(data):
    clf = joblib.load("dbscan_model.sav")
    return clf.predict(data)