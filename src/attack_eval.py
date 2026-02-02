import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score

model = joblib.load("model/random_forest.pkl")

X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X = X / 255.0
y = y.astype(int)

def random_sign_attack(X, epsilon):
    noise = np.sign(np.random.randn(*X.shape))
    return np.clip(X + epsilon * noise, 0, 1)

X_test = X[:8000]
y_test = y[:8000]

epsilons = [0.0, 0.05, 0.1, 0.2, 0.3]

print("\nRandom Forest robustness under random sign attack")
print("------------------------------------------------")

for eps in epsilons:
    X_adv = random_sign_attack(X_test, eps)
    preds = model.predict(X_adv)
    acc = accuracy_score(y_test, preds)
    print(f"epsilon={eps:.2f}  accuracy={acc:.4f}")
