# AdversarialRobustnessMNIST
Investigates the robustness of various supervised learning models against adversarial attacks on the MNIST dataset.
This project has evolved from a notebook-centric experiment into a deployable ML service with training scripts, a REST API, and Docker containerisation.
## Features
Train a Random Forest model on MNIST
- Evaluate model robustness under random adversarial perturbations
- Deploy the inference model as a REST API using FastAPI
- Containerise the service with Docker
- Clean project structure with scripts and evaluation pipelines

## Repository Structure
```bash
AdversarialRobustnessMNIST/
│
├── src/
│   ├── app.py                ← FastAPI server
│   ├── train_rf.py           ← Train Random Forest
│   └── attack_eval.py        ← Offline robustness evaluation
│
├── model/
│   └── model.pkl             ← Saved artefact (ignored in Git)
│
├── Dockerfile                ← Container build config
├── requirements.txt          ← Python dependencies
├── README.md                 ← This file
└── Adversarial.ipynb         ← Original experimental notebook

```
## Background
Adversarial attacks small, intentional perturbations to input data can cause machine learning models to make wrong predictions.    
This repository:

- Compares classifiers under adversarial pressure
- Builds a reproducible pipeline for model training and evaluation
- Demonstrates a deployable service with containerisation
## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/adityaMachal/AdversarialRobustnessMNIST.git
   cd AdversarialRobustnessMNIST
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   venv/Scripts/activate
   ```   
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Training the Model
```bash
python src/train_rf.py
```
## Run the API
1. Start FastAPI:
   ```bash
   uvicorn src.app:app --reload
   ```
2. Navigate to:
   ```bash
     http://127.0.0.1:8000/docs
   ```
Use the interactive docs to POST to ```/predict```.
## Run with Docker
Build the docker image:
```bash
  docker build -t rf-api:1.0 .
```
Run the container:
```bash
docker run -p 8000:8000 rf-api:1.0
```
You can then visit the same docs at:
```bash
http://localhost:8000/docs
```
## Adversarial Evaluation
The ```attack_eval.py``` script generates adversarial perturbations (random sign attack) and measures accuracy under different noise levels:
```bash
python src/attack_eval.py
```
This produces an accuracy summary for increasing perturbation strengths.

## Contributing
Contributions are welcome! Please:
- Fork the repository
- Create a feature branch
- Add meaningful commits
- Open a Pull Request
