# 🛡️ AdversarialRobustnessMNIST

Investigating the robustness of supervised learning models against adversarial attacks on the MNIST handwritten-digit dataset.

This project benchmarks **five classifiers** — from traditional ML to deep learning — under adversarial perturbations of increasing strength, and packages the best-performing model as a deployable REST API.

---

## 📌 Features

- **Multi-model comparison** — Logistic Regression, Decision Tree, Random Forest, CNN, and KNN trained and evaluated side by side
- **Adversarial attack simulation** — FGSM and gradient-approximation attacks at multiple perturbation strengths (ε = 0.0 – 0.3)
- **Visual analysis** — Clean vs adversarial accuracy plots for every model, plus a combined robustness comparison chart
- **REST API** — FastAPI-powered inference endpoint for real-time digit prediction
- **Docker support** — Containerised deployment with a single command

---

## 🏗️ Repository Structure

```
AdversarialRobustnessMNIST/
│
├── Adversarial.ipynb         ← Full experimental notebook (all 5 models)
│
├── src/
│   ├── app.py                ← FastAPI inference server
│   ├── train_rf.py           ← Random Forest training script
│   ├── attack_eval.py        ← Offline adversarial robustness evaluation
│   └── predict.py            ← Quick prediction smoke test
│
├── model/
│   └── random_forest.pkl     ← Trained model artefact (git-ignored)
│
├── dockerfile                ← Container build configuration
├── requirements.txt          ← Python dependencies
├── LICENSE                   ← MIT License
└── README.md                 ← This file
```

---

## 🧠 Models & Experiments

### Models Trained

| # | Model | Clean Accuracy | Key Characteristics |
|---|-------|---------------|---------------------|
| 1 | **Logistic Regression** | 92.58% | Linear classifier; trained with `max_iter=1000` |
| 2 | **Decision Tree** | 88.18% | Non-linear; `max_depth=20` to control overfitting |
| 3 | **Random Forest** | 97.04% | Ensemble of 100 trees; strong baseline |
| 4 | **CNN (Keras)** | 99.21% | 2 Conv layers + Dense + Dropout; trained for 10 epochs |
| 5 | **KNN** | 97.05% | Instance-based; `k=3` neighbours |

### Adversarial Attack Methods

Each model is evaluated under an attack strategy suited to its architecture:

- **Logistic Regression** — Analytical FGSM using model coefficients (`model.coef_`) to compute exact gradient direction
- **Decision Tree & Random Forest** — Gradient-free perturbation using random sign directions (trees have no differentiable loss surface)
- **CNN** — True FGSM via `tf.GradientTape`, computing input gradients of the cross-entropy loss
- **KNN** — Heuristic FGSM that perturbs inputs towards wrong-class neighbours

### Robustness Results

Accuracy (%) under FGSM attack across increasing perturbation strength ε:

| ε | Logistic Reg. | Decision Tree | Random Forest | CNN | KNN |
|---|--------------|---------------|---------------|-----|-----|
| 0.00 | 92.58 | 88.18 | 97.04 | 99.21 | 97.05 |
| 0.05 | 24.55 | 51.00 | 71.15 | 96.68 | 95.51 |
| 0.10 | 0.64 | 31.89 | 49.90 | 90.01 | 94.25 |
| 0.15 | 0.00 | 23.57 | 36.62 | 76.24 | 93.21 |
| 0.20 | 0.00 | 18.61 | 28.38 | 56.05 | 92.43 |
| 0.25 | 0.00 | 15.87 | 23.40 | 37.38 | 91.83 |
| 0.30 | 0.00 | 14.14 | 21.60 | 25.85 | 91.31 |

### Key Takeaways

- **KNN is the most robust** — retains over 91% accuracy even at ε = 0.3
- **CNN achieves the highest clean accuracy** (99.21%) but degrades significantly under strong perturbations
- **Logistic Regression is the most vulnerable** — accuracy drops to near 0% at ε ≥ 0.1
- **Ensemble methods (Random Forest)** provide better resilience than single Decision Trees
- Adversarial robustness and clean accuracy are often at odds — no single model dominates both

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/adityaMachal/AdversarialRobustnessMNIST.git
   cd AdversarialRobustnessMNIST
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux / macOS
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏋️ Training the Model

Train the Random Forest classifier on MNIST:

```bash
python src/train_rf.py
```

This downloads MNIST via `fetch_openml`, trains a 100-tree Random Forest, prints test accuracy, and saves the model to `model/random_forest.pkl`.

---

## 🔬 Adversarial Evaluation

Run the offline robustness evaluation against random sign perturbations:

```bash
python src/attack_eval.py
```

**Sample output:**
```
Random Forest robustness under random sign attack
------------------------------------------------
epsilon=0.00  accuracy=0.9704
epsilon=0.05  accuracy=0.9352
epsilon=0.10  accuracy=0.8769
epsilon=0.20  accuracy=0.7310
epsilon=0.30  accuracy=0.5874
```

---

## 🌐 Running the API

### Local

1. **Start the FastAPI server:**
   ```bash
   uvicorn src.app:app --reload
   ```

2. **Open the interactive docs:**
   ```
   http://127.0.0.1:8000/docs
   ```

3. **Make a prediction** — POST a flat list of 784 pixel values (0–1) to `/predict`:
   ```bash
   curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"data": [0.0, 0.0, ..., 0.0]}'
   ```

   **Response:**
   ```json
   {"prediction": 7}
   ```

### Docker

```bash
# Build
docker build -t rf-api:1.0 .

# Run
docker run -p 8000:8000 rf-api:1.0
```

Then visit `http://localhost:8000/docs` for the Swagger UI.

---

## 📓 Notebook

The full experimental workflow lives in [`Adversarial.ipynb`](Adversarial.ipynb):

1. **Data loading & visualisation** — raw MNIST binary files parsed and displayed
2. **Model training** — all five classifiers trained from scratch
3. **Adversarial attack generation** — per-model attack functions with configurable ε
4. **Per-model evaluation** — clean vs adversarial accuracy plots
5. **Cross-model comparison** — unified robustness comparison chart

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| ML / Data | NumPy, pandas, scikit-learn, TensorFlow / Keras |
| Visualisation | Matplotlib, Seaborn |
| API | FastAPI, Uvicorn |
| Infrastructure | Docker, Joblib |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---
