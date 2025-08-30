
import os, re, pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(__file__))
RAW = os.path.join(ROOT, "data", "raw", "sample_emails.csv")
OUT_DIR = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

def strip_html(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Remove HTML and decode entities
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(" ")
    # Normalize whitespace & lowercase
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def main():
    df = pd.read_csv(RAW)
    # Basic clean
    df["subject_clean"] = df["subject"].map(strip_html)
    df["body_clean"] = df["body"].map(strip_html)
    df["text"] = (df["subject_clean"] + " " + df["body_clean"]).str.strip()
    df = df[["text","label"]]

    # Split (stratified)
    train_df, test_df = train_test_split(
        df, test_size=0.25, random_state=42, stratify=df["label"]
    )

    train_out = os.path.join(OUT_DIR, "train.csv")
    test_out = os.path.join(OUT_DIR, "test.csv")
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    print(f"✅ Wrote {len(train_df)} rows to {train_out}")
    print(f"✅ Wrote {len(test_df)} rows to {test_out}")
    print(train_df['label'].value_counts().to_frame('train_counts'))
    print(test_df['label'].value_counts().to_frame('test_counts'))

if __name__ == "__main__":
    main()
