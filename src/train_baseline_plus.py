# src/train_baseline_plus.py
import os, sys, joblib, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# Make 'src' importable and bring in our helpers
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
from features import to_feature_frame  # IMPORTANT: import, don't define inline

TRAIN = os.path.join(ROOT, "data", "processed", "train.csv")
TEST  = os.path.join(ROOT, "data", "processed", "test.csv")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    train_df = pd.read_csv(TRAIN)
    test_df  = pd.read_csv(TEST)

    X_train = pd.DataFrame({"text": train_df["text"].tolist()})
    y_train = train_df["label"].values
    X_test  = pd.DataFrame({"text": test_df["text"].tolist()})
    y_test  = test_df["label"].values

    # Combine TF-IDF text features + engineered numeric URL features
    pre = ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1), "text"),
            ("urlfeat", Pipeline(steps=[
                ("map", FunctionTransformer(to_feature_frame, validate=False)),
                ("scale", StandardScaler(with_mean=False)),
            ]), "text"),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipe = Pipeline(steps=[
        ("prep", pre),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    # AUC
    label_map = {"phishing":1, "legit":0}
    y_test_bin = np.array([label_map[y] for y in y_test])
    y_proba = pipe.predict_proba(X_test)[:, 1]
    print(f"ROC-AUC: {roc_auc_score(y_test_bin, y_proba):.4f}")

    out = os.path.join(MODELS_DIR, "baseline_plus_v2.joblib")
    joblib.dump(pipe, out)
    print(f"✅ Saved model to {out}")

if __name__ == "__main__":
    main()
