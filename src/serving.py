from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load('models/model.pkl')

@app.post("/predict")
def predict(data: dict):
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}