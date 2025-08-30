# src/predict.py
import os, sys, joblib
from text_clean import simple_clean 


ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "baseline_lr_tfidf.joblib")

def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Train first: python src\\train_baseline.py")
        sys.exit(1)
    return joblib.load(MODEL_PATH)

def predict(text: str):
    model = load_model()
    proba = model.predict_proba([text])[0][1]   # probability of phishing
    label = "phishing" if proba >= 0.5 else "legit"
    return label, float(proba)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage:\n  python src\\predict.py \"<email subject + body>\"")
        sys.exit(0)
    text = sys.argv[1]
    label, p = predict(text)
    print(f"Prediction: {label} (phishing probability: {p:.3f})")
