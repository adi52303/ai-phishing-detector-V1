import re
import warnings
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

# Silence the “looks like a URL” BeautifulSoup warning
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# ---------- Candidate column names (case-insensitive) ----------
TEXT_CANDIDATES    = ["text", "body", "email_text", "message", "content", "email", "clean_text", "email text", "email body"]
SUBJECT_CANDIDATES = ["subject", "title", "email subject", "subject line"]
URL_CANDIDATES     = ["url", "urls", "link", "links"]
LABEL_CANDIDATES   = ["label", "is_phishing", "phishing", "class", "target", "y", "email type", "type", "category"]

# ---------- Label vocabulary (normalize to 1=phish, 0=ham) ----------
PHISH_VALUES = {
    "phish", "phishing", "phishing email", "spam", "malicious", "fraud",
    "1", "true", "yes"
}
HAM_VALUES = {
    "ham", "legit", "legitimate", "benign", "safe", "safe email",
    "0", "false", "no", "not_phish"
}

def _norm_col(s: str) -> str:
    """
    Normalize a column name: lowercase, collapse whitespace (incl. NBSP),
    and strip punctuation so 'Email Type' / 'email-type' / 'Email Type' all match.
    """
    s = (s or "").lower()
    s = s.replace("\xa0", " ")                 # NBSP -> space
    s = re.sub(r"\s+", " ", s).strip()         # collapse spaces
    s = re.sub(r"[^a-z0-9 ]+", "", s)          # keep letters/digits/spaces
    return s

def autodetect_column(cols, candidates):
    """
    Robust column autodetect:
      1) exact match on normalized form
      2) substring fallback (e.g., 'email text body' matches 'email text')
    """
    if not cols:
        return None
    norm_map = {_norm_col(c): c for c in cols if isinstance(c, str)}
    cand_norms = [_norm_col(c) for c in candidates]
    # exact match
    for cn in cand_norms:
        if cn in norm_map:
            return norm_map[cn]
    # substring fallback
    for cn in cand_norms:
        for k, orig in norm_map.items():
            if cn and cn in k:
                return orig
    return None

def normalize_label(val):
    """
    Convert various label spellings/encodings to {1=phish, 0=ham}.
    Returns np.nan if unknown.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in PHISH_VALUES:
        return 1
    if s in HAM_VALUES:
        return 0
    # numeric fallback
    try:
        f = float(s)
        if f == 1.0:
            return 1
        if f == 0.0:
            return 0
    except Exception:
        pass
    return np.nan

_HTML_TAG = re.compile(r"<[^>]+>")

def clean_text(s):
    """
    Strip HTML, lowercase, collapse whitespace.
    Safe for NaN and non-string inputs.
    """
    if pd.isna(s):
        return ""
    s = str(s)
    try:
        s = BeautifulSoup(s, "html.parser").get_text(" ")
    except Exception:
        s = _HTML_TAG.sub(" ", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s
