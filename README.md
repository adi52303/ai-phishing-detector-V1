🛡️ AI Phishing Email Detector

📌 Project Overview

Phishing remains one of the most dangerous cybersecurity threats, tricking users into revealing sensitive information via fraudulent emails.

This project implements an AI-powered phishing email detector, trained on 530k+ real and phishing emails. The pipeline covers the end-to-end ML lifecycle:

Data preprocessing & cleaning

Feature engineering (TF-IDF + URL/domain heuristics)

Model training & evaluation (Logistic Regression)

Deployment as an interactive Streamlit web app

✅ Achieves 98.4% accuracy and ~100% recall on phishing emails.

📊 Datasets

Enron Email Dataset (Kaggle)

~500,000 legitimate emails from Enron employees

Used as the source of non-phishing (ham) examples

Phishing Emails Dataset (Kaggle)

~18,000 phishing emails labeled malicious

Used as the phishing class for training

⚠️ Note:

Datasets are not included in this repo due to GitHub’s file size limit (100 MB).

To retrain the model, please download datasets directly from Kaggle.

The trained model (tfidf_logreg.joblib) is already included for demo use.

🧠 Model Pipeline
graph TD
A[Raw Data (Enron + Phishing)] --> B[Preprocessing & Cleaning]
B --> C[Feature Engineering: TF-IDF + URL Features]
C --> D[Train Logistic Regression]
D --> E[Evaluation: Accuracy, Recall, F1]
E --> F[Save Model: tfidf_logreg.joblib]
F --> G[Streamlit App for Deployment]

📈 Results

Accuracy: 98.4%

Recall (phishing): ~100%

Precision (phishing): 94%

Confusion Matrix:




🖥️ Streamlit App

The model is deployed as an interactive Streamlit app.
📌 👉 Live Demo Here- https://ai-phishing-detector-v1-pqj83jt538g3fbhhnqxq2x.streamlit.app/

Features:

Paste an email → get instant prediction (Phishing / Legit)

Adjustable decision threshold slider

Explainability panel showing: suspicious words, domains, URLs, etc.



⚙️ How to Run Locally

Clone this repo:

git clone https://github.com/<your-username>/ai-phishing-detector-V1.git
cd ai-phishing-detector-V1


Install dependencies:

pip install -r requirements.txt


Run Streamlit app:

streamlit run app/app.py


Open in your browser at http://localhost:8501/.

🔮 Future Improvements

Integrate transformer models (e.g., BERT, DistilBERT) for better context understanding

Deploy as a REST API (FastAPI) for production use

Add real-time phishing detection from email servers

📜 License

This project is licensed under the MIT License.
