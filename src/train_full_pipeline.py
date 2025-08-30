import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from .utils import (
    TEXT_CANDIDATES, SUBJECT_CANDIDATES, LABEL_CANDIDATES,
    autodetect_column, normalize_label, clean_text
)

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def _clean_or_fallback(series):
    """Clean text; if cleaning produces empty, fall back to original raw string."""
    raw = series.fillna("").astype(str)
    cleaned = raw.map(clean_text)
    empty_mask = cleaned.str.strip() == ""
    cleaned.loc[empty_mask] = raw.loc[empty_mask]
    return cleaned

def normalize_csvs(raw_dir, out_path):
    """
    Read all CSVs in raw_dir and produce a unified file with columns ['text','label'].
    - Fast-path passthrough for files that *already* have text+label.
    - Schema-map for known tricky files.
    - Autodetect as a fallback for others.
    """
    SCHEMA_MAP = {
        # Explicit mappings (filename -> column mapping).
        # If a file has no label column but is all ham, set label to a constant 0.
        "Phishing_Email.csv": {"text": "Email Text", "label": "Email Type"},
        "emails.csv":         {"text": "message",    "label": 0},  # legit-only file you have
        # Add more here as needed:
        # "legit_emails.csv": {"text": "Body", "label": 0},
    }

    frames = []
    for name in os.listdir(raw_dir):
        if not name.lower().endswith(".csv"):
            continue
        path = os.path.join(raw_dir, name)
        try:
            df = pd.read_csv(path)
            cols = list(df.columns)

            # -------- Fast-path: already has text+label (any capitalization) --------
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

            # -------- Schema-map or autodetect --------
            schema = SCHEMA_MAP.get(name)
            if schema:
                text_col = schema.get("text")
                sub_col  = schema.get("subject")
                lab_col  = schema.get("label")
            else:
                text_col = autodetect_column(cols, TEXT_CANDIDATES)
                sub_col  = autodetect_column(cols, SUBJECT_CANDIDATES)
                lab_col  = autodetect_column(cols, LABEL_CANDIDATES)

            out = pd.DataFrame()

            # text/subject
            if text_col and text_col in df.columns:
                out["body"] = _clean_or_fallback(df[text_col])
            else:
                out["body"] = ""
            if sub_col and sub_col in df.columns:
                out["subject"] = _clean_or_fallback(df[sub_col])
            else:
                out["subject"] = ""
            out["text"] = (out["subject"] + " " + out["body"]).astype(str).str.strip()

            # label
            if isinstance(lab_col, (int, float)) and not isinstance(lab_col, bool):
                out["label"] = int(lab_col)  # constant label
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
    # final safety
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
    """Keep all phishing (label=1), sample ham (label=0) up to ham_per_phish per phishing."""
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

def build_pipeline(model=None):
    """Return a TF-IDF + classifier pipeline."""
    if model is None:
        model = LogisticRegression(
            max_iter=200,
            class_weight="balanced",
            random_state=42
        )
    # Tip: if RAM is tight with huge data, add min_df=2 or 3 to shrink the vocab
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2))),  # add min_df=2 if needed
        ("clf", model)
    ])

def main():
    norm_file = os.path.join(PROCESSED_DIR, "normalized.csv")
    df = normalize_csvs(RAW_DIR, norm_file)

    # Balance big ham vs phish before splitting
    df = balance_dataset(df, ham_per_phish=3)

    print("\n📊 Label distribution (post-balance):")
    print(df["label"].value_counts())

    # Split train/val/test
    train_df, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=42)

    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

    print(f"✅ Split sizes → train:{len(train_df)} val:{len(val_df)} test:{len(test_df)}")

    # Train baseline model
    pipe = build_pipeline()
    pipe.fit(train_df["text"], train_df["label"])

    # Validation report
    preds = pipe.predict(val_df["text"])
    print("\n📊 Validation Report:")
    print(classification_report(val_df["label"], preds, digits=3))

    # Save model
    model_path = os.path.join(MODEL_DIR, "tfidf_logreg.joblib")
    joblib.dump(pipe, model_path)
    print(f"✅ Saved model: {model_path}")

    # Save TF-IDF vocab for inspection
    vocab = pipe.named_steps["tfidf"].get_feature_names_out()
    pd.Series(vocab).to_csv(os.path.join(MODEL_DIR, "tfidf_features.csv"), index=False)

    # Test evaluation
    test_preds = pipe.predict(test_df["text"])
    print("\n🧪 Test Report:")
    print(classification_report(test_df["label"], test_preds, digits=3))
    print("Confusion Matrix:")
    print(confusion_matrix(test_df["label"], test_preds))

    # Save test metrics to file
    report = classification_report(test_df["label"], test_preds, digits=3, output_dict=True)
    pd.DataFrame(report).to_csv(os.path.join(PROCESSED_DIR, "test_report.csv"))
    print(f"✅ Test metrics saved to {os.path.join(PROCESSED_DIR, 'test_report.csv')}")

if __name__ == "__main__":
    main()
