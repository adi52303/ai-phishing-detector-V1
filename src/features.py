# src/train_full_pipeline.py
import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse as sp
from sklearn.preprocessing import MaxAbsScaler

# ---- our engineered features ----
from .features import to_feature_frame

# ---- your existing helpers (unchanged) ----
from .utils import (
    TEXT_CANDIDATES, SUBJECT_CANDIDATES, LABEL_CANDIDATES,
    autodetect_column, normalize_label, clean_text
)

# ---- header candidates we want to ingest ----
FROM_CANDIDATES = ["from", "from_", "sender", "from address", "from_address", "email_from"]
TO_CANDIDATES   = ["to", "recipient", "email_to", "to_address"]

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------
# Normalization helpers
# ---------------------------
def _clean_or_fallback(series):
    raw = series.fillna("").astype(str)
    cleaned = raw.map(clean_text)
    empty_mask = cleaned.str.strip() == ""
    cleaned.loc[empty_mask] = raw.loc[empty_mask]
    return cleaned


def normalize_csvs(raw_dir, out_path):
    SCHEMA_MAP = {
        "Phishing_Email.csv": {"text": "Email Text", "label": "Email Type"},
        "emails.csv":         {"text": "message",    "label": 0},
    }

    frames = []
    for name in os.listdir(raw_dir):
        if not name.lower().endswith(".csv"):
            continue
        path = os.path.join(raw_dir, name)
        try:
            df = pd.read_csv(path)
            cols = list(df.columns)

            lower_cols = {c.lower(): c for c in cols}
            if {"text", "label"}.issubset(lower_cols.keys()):
                df2 = df.rename(columns={lower_cols["text"]: "text", lower_cols["label"]: "label"})
                df2 = df2.dropna(subset=["text", "label"]).copy()
                df2["text"] = df2["text"].astype(str).str.strip()
                df2 = df2[df2["text"] != ""]
                df2["label"] = df2["label"].astype(int)
                frames.append(df2[["text", "label"]])
                print(f"[OK] {name} (passthrough) → {len(df2)} rows")
                continue

            schema = SCHEMA_MAP.get(name)
            if schema:
                text_col = schema.get("text")
                sub_col  = schema.get("subject")
                lab_col  = schema.get("label")
            else:
                text_col = autodetect_column(cols, TEXT_CANDIDATES)
                sub_col  = autodetect_column(cols, SUBJECT_CANDIDATES)
                lab_col  = autodetect_column(cols, LABEL_CANDIDATES)

            # detect From/To columns
            from_col = autodetect_column(cols, FROM_CANDIDATES)
            to_col   = autodetect_column(cols, TO_CANDIDATES)

            out = pd.DataFrame()

            if text_col and text_col in df.columns:
                out["body"] = _clean_or_fallback(df[text_col])
            else:
                out["body"] = ""
            if sub_col and sub_col in df.columns:
                out["subject"] = _clean_or_fallback(df[sub_col])
            else:
                out["subject"] = ""

            out["from_addr"] = _clean_or_fallback(df[from_col]) if (from_col and from_col in df.columns) else ""
            out["to_addr"]   = _clean_or_fallback(df[to_col])   if (to_col   and to_col   in df.columns)   else ""

            # Build a single text field that includes Subject / From / To / Body
            out["text"] = (
                "Subject: " + out["subject"].astype(str).str.strip() + "\n" +
                "From: "    + out["from_addr"].astype(str).str.strip() + "\n" +
                "To: "      + out["to_addr"].astype(str).str.strip() + "\n\n" +
                out["body"].astype(str).str.strip()
            ).str.strip()

            if isinstance(lab_col, (int, float)) and not isinstance(lab_col, bool):
                out["label"] = int(lab_col)
            elif lab_col == 0:
                out["label"] = 0
            elif lab_col == 1:
                out["label"] = 1
            elif isinstance(lab_col, str) and lab_col in df.columns:
                out["label"] = df[lab_col].map(normalize_label)
            else:
                print(f"[SKIP] {name}: no usable label mapping (lab_col={lab_col})")
                print(f"[DBG] {name} columns: {cols}")
                continue

            before = len(out)
            out = out.dropna(subset=["label"]).reset_index(drop=True)

            out["text"] = out["text"].astype(str)
            text_norm = out["text"].str.strip().str.lower()
            out = out[(text_norm.notna()) & (text_norm != "") & (text_norm != "nan")]
            after_text = len(out)

            if out.empty:
                total = len(df)
                lbl_col = lab_col if isinstance(lab_col, str) else f"(const={lab_col})"
                txt_col = text_col or "(none)"
                sub_col_name = sub_col or "(none)"
                unknown_label = (
                    df[lab_col].map(normalize_label).isna().sum()
                    if isinstance(lab_col, str) and lab_col in df.columns
                    else f"const={lab_col}"
                )
                print(f"[SKIP] {name}: no usable rows after cleaning")
                print(f"[DBG] {name}: rows={total}, label_col={lbl_col}, text_col={txt_col}, subject_col={sub_col_name}, "
                      f"after_clean={after_text}/{before}, unknown_label≈{unknown_label}")
                try:
                    if isinstance(lab_col, str) and lab_col in df.columns:
                        print("[DBG] label head value_counts():")
                        print(df[lab_col].head(100).value_counts(dropna=False).head(10))
                except Exception:
                    pass
                continue

            frames.append(out[["text", "label"]])
            print(f"[OK] {name} → {len(out)} rows")

        except Exception as e:
            print(f"[ERROR] {name}: {e}")

    if not frames:
        raise SystemExit("No valid rows found across CSVs. Check mappings & columns.")

    all_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["text"])
    all_df["text"] = all_df["text"].astype(str)
    text_norm = all_df["text"].str.strip().str.lower()
    all_df = all_df[(text_norm.notna()) & (text_norm != "") & (text_norm != "nan")]

    all_df.to_csv(out_path, index=False)
    print(f"✅ Saved normalized file: {out_path} ({len(all_df)} rows)")

    print("\n🔍 Sample normalized rows:")
    print(all_df.head(5).to_string(index=False))
    print("Any NaN in text?", int(all_df["text"].isna().sum()))
    print("Empty text rows:", int((all_df['text'].str.strip() == '').sum()))
    return all_df


def balance_dataset(df, ham_per_phish=3, random_state=42):
    phish = df[df["label"] == 1]
    ham = df[df["label"] == 0]
    if len(phish) == 0:
        print("⚠️ No phishing rows found; skipping balancing.")
        return df
    max_ham = min(len(ham), len(phish) * ham_per_phish)
    ham_sampled = ham.sample(n=max_ham, random_state=random_state) if max_ham > 0 else ham
    out = pd.concat([phish, ham_sampled], ignore_index=True)
    out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    print(f"🔧 Balancing: phish={len(phish)} ham_used={len(ham_sampled)} total={len(out)} (ratio≈1:{ham_per_phish})")
    return out


# ---------------------------
# Hybrid feature builder: TF-IDF + Engineered URL features
# ---------------------------
class UrlFeaturizer(BaseEstimator, TransformerMixin):
    """Turns raw text -> numeric engineered URL features (csr_matrix)."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X can be list/Series/ndarray; convert to pandas Series of strings
        if isinstance(X, pd.Series):
            texts = X
        else:
            texts = pd.Series(X, dtype="object").astype(str)
        df = to_feature_frame(texts)
        # ensure float32 and convert to sparse for FeatureUnion
        mat = sp.csr_matrix(df.values.astype("float32"))
        return mat


def build_pipeline():
    tfidf = ("tfidf", TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,                 # trims 1-off tokens → stabler LR
    ))

    url_block = ("url", Pipeline([
        ("feats", UrlFeaturizer()),
        ("scale", MaxAbsScaler())  # put numeric features on similar scale
    ]))

    features = FeatureUnion([
        tfidf,
        url_block,
    ])

    clf = LogisticRegression(
        solver="saga",            # good for large sparse
        max_iter=2000,            # avoid convergence warnings
        class_weight="balanced",
        C=0.5,                    # tighter weights to reduce overfitting
        random_state=42,
        n_jobs=-1
    )

    return Pipeline([
        ("features", features),
        ("clf", clf),
    ])


def main():
    norm_file = os.path.join(PROCESSED_DIR, "normalized.csv")
    df = normalize_csvs(RAW_DIR, norm_file)

    df = balance_dataset(df, ham_per_phish=3)

    print("\n📊 Label distribution (post-balance):")
    print(df["label"].value_counts())

    # Split
    train_df, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=42)

    # Save splits (optional)
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
    print(f"✅ Split sizes → train:{len(train_df)} val:{len(val_df)} test:{len(test_df)}")

    # Train hybrid model
    pipe = build_pipeline()
    pipe.fit(train_df["text"], train_df["label"])

    # Validation
    val_pred = pipe.predict(val_df["text"])
    print("\n📊 Validation Report:")
    print(classification_report(val_df["label"], val_pred, digits=3))

    # --- derive & save a recommended probability threshold ---
    val_probs = pipe.predict_proba(val_df["text"])[:, 1]
    prec, rec, thr = precision_recall_curve(val_df["label"], val_probs)
    f1 = (2 * prec * rec) / (prec + rec + 1e-9)
    best_idx = f1.argmax()
    best_thr = float(thr[best_idx]) if best_idx < len(thr) else 0.5
    ap = float(average_precision_score(val_df["label"], val_probs))
    print(f"\n🔧 Suggested threshold (max F1): {best_thr:.3f} | Val AP: {ap:.3f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, 'threshold.txt'), 'w') as f:
        f.write(str(best_thr))

    # (optional) persist a couple of metrics
    with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w') as f:
        json.dump({
            'suggested_threshold_f1': best_thr,
            'average_precision': ap
        }, f, indent=2)

    # Save model
    model_path = os.path.join(MODEL_DIR, "tfidf_logreg.joblib")
    joblib.dump(pipe, model_path)
    print(f"✅ Saved model: {model_path}")

    # Test
    test_pred = pipe.predict(test_df["text"])
    print("\n🧪 Test Report:")
    print(classification_report(test_df["label"], test_pred, digits=3))
    print("Confusion Matrix:")
    print(confusion_matrix(test_df["label"], test_pred))


if __name__ == "__main__":
    main()
