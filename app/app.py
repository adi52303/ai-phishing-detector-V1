# app/app.py
import os
import sys
import joblib
import streamlit as st
import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
if ROOT not in sys.path:
    sys.path.append(ROOT)


from src.features import extract_url_features  


try:
    
    from src.train_full_pipeline import UrlFeaturizer  
except Exception:
    
    from sklearn.base import BaseEstimator, TransformerMixin
    from scipy import sparse as sp
    import pandas as _pd
    from src.features import to_feature_frame as _to_feature_frame

    class UrlFeaturizer(BaseEstimator, TransformerMixin):  # noqa: F811
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            texts = X if isinstance(X, _pd.Series) else _pd.Series(X, dtype="object").astype(str)
            df = _to_feature_frame(texts)
            return sp.csr_matrix(df.values.astype("float32"))

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(ROOT, "models", "tfidf_logreg.joblib")
THR_PATH   = os.path.join(ROOT, "models", "threshold.txt")

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Phishing Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>🛡️ AI Phishing Detector</h1>",
    unsafe_allow_html=True
)
st.write(
    "Paste an email or fill the fields. The model predicts **Phishing** or **Legit** "
    "with probability and shows feature signals."
)

# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=True)
def load_model(path: str):
    if not os.path.exists(path):
        st.error("❌ Model not found. Train it with: `python -m src.train_full_pipeline`")
        st.stop()
    return joblib.load(path)

def load_threshold(default_val: float = 0.36) -> float:
    try:
        with open(THR_PATH) as f:
            return float(f.read().strip())
    except Exception:
        return default_val

def build_raw_text(subject: str, from_addr: str, to_addr: str, body: str) -> str:
    subject = (subject or "").strip()
    from_addr = (from_addr or "").strip()
    to_addr = (to_addr or "").strip()
    body = (body or "").strip()
    return f"""Subject: {subject}
From: {from_addr}
To: {to_addr}

{body}""".strip()

model = load_model(MODEL_PATH)
suggested_thr = load_threshold(0.36)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: threshold control
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    threshold = st.slider(
        "Decision threshold (lower = stricter, higher = lenient)",
        min_value=0.05, max_value=0.95, value=float(suggested_thr), step=0.01,
        help="Default is the best-F1 threshold learned on validation data."
    )
    st.caption(f"Suggested threshold from validation: **{suggested_thr:.3f}**")

# ─────────────────────────────────────────────────────────────────────────────
# Input mode
# ─────────────────────────────────────────────────────────────────────────────
mode = st.radio("Input mode", ["Structured (fields)", "Raw paste"], horizontal=True)

if mode == "Structured (fields)":
    col1, col2 = st.columns(2)
    with col1:
        subject = st.text_input("Subject", value="")
        from_addr = st.text_input("From", value="")
    with col2:
        to_addr = st.text_input("To", value="")
    body = st.text_area("Body", height=220, value="")
    raw_text = build_raw_text(subject, from_addr, to_addr, body)
else:
    raw_text = st.text_area(
        "✉️ Email text (you can paste headers + body)",
        height=240,
        placeholder="Subject: ...\nFrom: ...\nTo: ...\n\nBody..."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────────────────────────────────────
if st.button("🔍 Analyze", type="primary", use_container_width=True):
    if not raw_text or not raw_text.strip():
        st.warning("⚠️ Please provide email content.")
        st.stop()

    try:
        proba = float(model.predict_proba([raw_text])[:, 1][0])  # P(phishing)
        label = "PHISHING" if proba >= threshold else "LEGIT"

        # Pipeline debugging info
        steps = list(getattr(model, "named_steps", {}).keys())
        st.caption(f"pipeline steps: {steps}")

        features_step = model.named_steps.get("features")
        clf = model.named_steps.get("clf")

        try:
            if features_step is not None and hasattr(features_step, "transformer_list"):
                sub_names = [name for name, _ in features_step.transformer_list]
                st.caption(f"feature union: {sub_names}")
                tfidf = dict(features_step.transformer_list).get("tfidf")
                if tfidf is not None and hasattr(tfidf, "vocabulary_"):
                    tfidf_vocab_size = len(getattr(tfidf, "vocabulary_", {}) or {})
                    st.caption(f"TF-IDF vocab size: {tfidf_vocab_size}")
        except Exception as ex:
            st.caption(f"(debug) feature union inspection failed: {ex}")

        if clf is not None and hasattr(clf, "coef_"):
            st.caption(f"coef L2 norm: {np.linalg.norm(clf.coef_):.6f}")

        # Label & probability
        if label == "PHISHING":
            st.markdown(
                f"<h2 style='color:#C0392B;'>🚨 Prediction: {label}</h2>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<h2 style='color:#27AE60;'>✅ Prediction: {label}</h2>",
                unsafe_allow_html=True
            )

        st.progress(min(max(proba, 0.0), 1.0))
        st.metric("Phishing Probability", f"{proba:.3f}")
        st.caption("Tip: adjust the threshold in the sidebar to tune sensitivity.")

        # ─────────────────────────────────────────────────────────────────
        # Explainability (feature signals) — aligned with src.features
        # ─────────────────────────────────────────────────────────────────
        with st.expander("📊 Why this decision? (feature signals)"):
            feat = extract_url_features(raw_text)

            urls_found = int(feat.get("num_urls", 0))
            emails_found = int(feat.get("num_emails", 0))
            safe_domains = int(feat.get("num_safe_domains", 0))
            suspicious_tlds = int(feat.get("num_suspicious_tlds", 0))
            kw_total = int(feat.get("kw_total", 0))
            kw_urg = int(feat.get("kw_urgency", 0))
            kw_sec = int(feat.get("kw_security", 0))
            kw_fin = int(feat.get("kw_financial", 0))

            bullets = []
            bullets.append(f"🔗 URLs found: **{urls_found}**")
            bullets.append("✉️ Email headers detected" if emails_found else "❌ No email headers detected")
            bullets.append("✅ Safe domain match" if safe_domains else "❌ No safe domain match")
            bullets.append("⚠️ Suspicious TLDs present" if suspicious_tlds else "✅ No suspicious TLDs")

            if kw_total:
                parts = []
                if kw_urg: parts.append("urgency")
                if kw_sec: parts.append("security")
                if kw_fin: parts.append("financial")
                bullets.append("⚠️ Contains phishing words (" + ", ".join(parts) + ")")
            else:
                bullets.append("✅ No strong phishing keywords")

            bullets.append("⚠️ Link text → URL mismatch" if feat.get("mismatch_display_href", 0) else "✅ No display/href mismatch")
            bullets.append("⚠️ ‘@’ found in URL path" if feat.get("at_in_path", 0) else "✅ No ‘@’ in URL path")
            bullets.append("⚠️ IP-based URL present" if feat.get("has_ip_url", 0) else "✅ No IP-based URLs")

            max_dot_count = int(feat.get("max_dot_count", 0))
            bullets.append("✅ Domain structure looks normal" if max_dot_count <= 3 else f"🔍 Deeply nested domain depth: {max_dot_count}")

            st.markdown("\n".join(f"- {b}" for b in bullets))

        with st.expander("Show raw text sent to model"):
            st.code(raw_text)

    except Exception as e:
        st.error("❌ Inference failed. Ensure the saved model is the HYBRID pipeline (TF-IDF + URL features).")
        st.exception(e)

st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("🔧 Model: TF-IDF + URL/Email Features + Logistic Regression • Built with Streamlit")
