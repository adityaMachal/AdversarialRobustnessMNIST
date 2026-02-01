from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model/random_forest.pkl")

class InputData(BaseModel):
    data: list

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/predict")
def predict(payload: InputData):
    arr = np.array(payload.data, dtype=np.float32).reshape(1, 784)
    pred = model.predict(arr)
    return {"prediction": int(pred[0])}
