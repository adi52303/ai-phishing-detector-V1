import os
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = RAW_DIR  # write outputs back into data/raw

def to_sample_format(df: pd.DataFrame, text_col: str, label_spec, out_path: Path):
    """
    Convert a dataframe to sample format: columns ['text', 'label'] with label 0/1.
    label_spec can be:
      - a string column name to map values from
      - an int (0 or 1) to set a constant label
    """
    out = pd.DataFrame()
    # text
    out["text"] = df[text_col].fillna("").astype(str)
    # label
    if isinstance(label_spec, str):
        # normalize label strings
        lab = df[label_spec].astype(str).str.strip().str.lower()
        phish_vals = {"phish","phishing","phishing email","spam","malicious","fraud","1","true","yes"}
        ham_vals   = {"ham","legit","legitimate","benign","safe","safe email","0","false","no","not_phish"}
        out["label"] = lab.map(lambda s: 1 if s in phish_vals else (0 if s in ham_vals else None))
    else:
        # constant label
        const = int(label_spec)
        out["label"] = const

    # clean up
    out = out.dropna(subset=["label"]).copy()
    out["label"] = out["label"].astype(int)
    out["text"] = out["text"].astype(str).str.strip()
    out = out[out["text"] != ""]
    out.drop_duplicates(subset=["text","label"], inplace=True)

    out.to_csv(out_path, index=False)
    print(f"✅ Wrote {out_path} (rows={len(out)})")
    print("Label counts:\n", out["label"].value_counts())
    return out

def main():
    # 1) Convert Phishing_Email.csv (Email Text, Email Type) -> sample format
    phish_path = RAW_DIR / "Phishing_Email.csv"
    if phish_path.exists():
        dfp = pd.read_csv(phish_path)
        to_sample_format(dfp, text_col="Email Text", label_spec="Email Type",
                         out_path=OUT_DIR / "phishing_as_sample.csv")
    else:
        print("⚠️ data/raw/Phishing_Email.csv not found—skipping phishing conversion")

    # 2) Optional: convert legit emails (emails.csv: message, no label) -> label=0
    legit_path = RAW_DIR / "emails.csv"  # change if your legit file is named differently
    if legit_path.exists():
        dfl = pd.read_csv(legit_path)
        # map message -> text, label=0 (ham)
        to_sample_format(dfl, text_col="message", label_spec=0,
                         out_path=OUT_DIR / "legit_as_sample.csv")
    else:
        print("ℹ️ data/raw/emails.csv not found—skipping legit conversion")

if __name__ == "__main__":
    main()
