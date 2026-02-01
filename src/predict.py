import joblib
import numpy as np

model = joblib.load("model/random_forest.pkl")

dummy = np.zeros((1, 784))
pred = model.predict(dummy)

print("Prediction:", pred)