import joblib

def predict(data):
    clf = joblib.load("kmeans_model.sav")
    return clf.predict(data)
