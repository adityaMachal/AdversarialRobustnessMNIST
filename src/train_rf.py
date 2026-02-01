import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from pathlib import Path

def main():
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    X = X / 255.0
    y = y.astype(int)

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    model = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("Test accuracy:", acc)

    Path("model").mkdir(exist_ok=True)
    joblib.dump(model, "model/random_forest.pkl")

if __name__ == "__main__":
    main()
