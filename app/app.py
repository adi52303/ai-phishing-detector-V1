# app/app.py
import os
import sys
import re
import joblib
import streamlit as st
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Project path setup
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.features import extract_url_features  
try:
    from src.train_full_pipeline import UrlFeaturizer  # noqa: F401
except Exception:
    from sklearn.base import BaseEstimator, TransformerMixin
    from scipy import sparse as sp
    from src.features import to_feature_frame as _to_feature_frame
    import pandas as _pd
    class UrlFeaturizer(BaseEstimator, TransformerMixin):  # noqa: F811
        def fit(self, X, y=None): return self
        def transform(self, X):
            texts = X if isinstance(X, _pd.Series) else _pd.Series(X, dtype="object").astype(str)
            df = _to_feature_frame(texts)
            return sp.csr_matrix(df.values.astype("float32"))

# ─────────────────────────────────────────────────────────────────────────────
# Constants & loaders
MODEL_PATH = os.path.join(ROOT, "models", "tfidf_logreg.joblib")
THR_PATH   = os.path.join(ROOT, "models", "threshold.txt")

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

def build_raw_text(subject, from_addr, to_addr, body) -> str:
    return f"""Subject: {(subject or '').strip()}
From: {(from_addr or '').strip()}
To: {(to_addr or '').strip()}

{(body or '').strip()}""".strip()

model = load_model(MODEL_PATH)
suggested_thr = load_threshold(0.36)

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit Page Config
st.set_page_config(page_title="AI Phishing Detector", page_icon="🛡️", layout="centered")
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🛡️ AI Phishing Detector</h1>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Input helpers
colA, colB = st.columns(2)
with colA:
    if st.button("Insert Legit Sample"):
        st.session_state["raw_text"] = """Subject: Meeting Reminder – Q3 Project Status Update
From: projectoffice@company.com
To: team@company.com

Hi Team, join via Zoom: https://zoom.us/j/93284726384"""
with colB:
    if st.button("Insert Phishing Sample"):
        st.session_state["raw_text"] = """Subject: Urgent — Verify Your Email
From: account-security@secure-mail-alerts.com
To: you@example.com

[accounts.google.com](http://bit.ly/4AbCdEf)"""

mode = st.radio("Input mode", ["Structured (fields)", "Raw paste"], horizontal=True)
if mode == "Structured (fields)":
    c1, c2 = st.columns(2)
    with c1:
        subject = st.text_input("Subject", value="")
        from_addr = st.text_input("From", value="")
    with c2:
        to_addr = st.text_input("To", value="")
    body = st.text_area("Body", height=220, value="")
    raw_text = build_raw_text(subject, from_addr, to_addr, body)
else:
    raw_text = st.text_area("✉️ Email text", height=240, key="raw_text")

# threshold slider in main body
threshold = st.slider("Decision threshold (lower = stricter, higher = lenient)",
                      min_value=0.05, max_value=0.95, value=float(suggested_thr), step=0.01)
st.caption(f"Suggested threshold from validation: **{suggested_thr:.3f}**")

dev_mode = st.checkbox("Developer mode (show model internals)", value=False)

# ─────────────────────────────────────────────────────────────────────────────
# Prediction
if st.button("🔍 Analyze", type="primary", use_container_width=True):
    if not raw_text.strip():
        st.warning("⚠️ Please provide email content.")
        st.stop()

    try:
        proba = float(model.predict_proba([raw_text])[:, 1][0])
        label = "PHISHING" if proba >= threshold else "LEGIT"

        # label display
        color = "#C0392B" if label == "PHISHING" else "#27AE60"
        icon = "🚨" if label == "PHISHING" else "✅"
        st.markdown(f"<h2 style='color:{color};'>{icon} Prediction: {label}</h2>", unsafe_allow_html=True)
        st.markdown(f"<div style='margin-top:-8px;margin-bottom:8px;font-size:13px;color:#5F6B7A;'>Threshold used: <b>{threshold:.2f}</b></div>", unsafe_allow_html=True)

        st.progress(min(max(proba, 0.0), 1.0))
        st.metric("Phishing Probability", f"{proba:.3f}")

        # developer mode internals
        if dev_mode:
            with st.expander("Model internals"):
                steps = list(getattr(model, "named_steps", {}).keys())
                st.caption(f"pipeline steps: {steps}")
                features_step = model.named_steps.get("features")
                clf = model.named_steps.get("clf")
                try:
                    if features_step is not None and hasattr(features_step, "transformer_list"):
                        sub_names = [n for n, _ in features_step.transformer_list]
                        st.caption(f"feature union: {sub_names}")
                        tfidf = dict(features_step.transformer_list).get("tfidf")
                        if tfidf is not None and hasattr(tfidf, "vocabulary_"):
                            st.caption(f"TF-IDF vocab size: {len(tfidf.vocabulary_)}")
                except Exception as ex:
                    st.caption(f"(debug fail: {ex})")
                if clf is not None and hasattr(clf, "coef_"):
                    st.caption(f"coef L2 norm: {np.linalg.norm(clf.coef_):.6f}")

        # explainability list (always visible)
        feat = extract_url_features(raw_text)
        bullets = []
        bullets.append(f"🔗 URLs found: **{int(feat.get('num_urls',0))}**")
        bullets.append("✉️ Email headers detected" if feat.get("num_emails",0) else "❌ No email headers")
        bullets.append("✅ Safe domain match" if feat.get("num_safe_domains",0) else "❌ No safe domain")
        bullets.append("⚠️ Suspicious TLDs" if feat.get("num_suspicious_tlds",0) else "✅ Normal TLDs")
        if feat.get("kw_total",0):
            parts=[]
            if feat.get("kw_urgency"): parts.append("urgency")
            if feat.get("kw_security"): parts.append("security")
            if feat.get("kw_financial"): parts.append("financial")
            bullets.append("⚠️ Contains phishing words ("+", ".join(parts)+")")
        else:
            bullets.append("✅ No strong phishing keywords")
        bullets.append("⚠️ Link text→URL mismatch" if feat.get("mismatch_display_href") else "✅ No display/href mismatch")
        bullets.append("⚠️ ‘@’ in URL path" if feat.get("at_in_path") else "✅ No ‘@’ in URL path")
        bullets.append("⚠️ IP-based URL" if feat.get("has_ip_url") else "✅ No IP-based URLs")
        bullets.append("🔍 Deep domain depth" if feat.get("max_dot_count",0)>3 else "✅ Domain structure normal")

        st.subheader("📊 Why this decision?")
        st.markdown("\n".join(f"- {b}" for b in bullets))

        # natural language explanation
        explanation = []
        if label == "PHISHING":
            if feat.get("kw_total",0): explanation.append("contains suspicious keywords")
            if feat.get("num_suspicious_tlds",0): explanation.append("uses suspicious domains")
            if feat.get("mismatch_display_href",0): explanation.append("link text doesn’t match its destination")
            if feat.get("has_ip_url",0): explanation.append("contains IP-based links")
            if not explanation: explanation.append("overall text patterns match phishing emails")
            st.info("⚠️ This email is likely phishing because it " + ", ".join(explanation) + ".")
        else:
            if feat.get("num_safe_domains",0): explanation.append("trusted domains detected")
            if not feat.get("kw_total",0): explanation.append("no suspicious keywords")
            if not feat.get("has_ip_url",0): explanation.append("no IP-based links")
            if not explanation: explanation.append("content looks typical for legitimate communication")
            st.success("✅ This email looks legitimate because " + ", ".join(explanation) + ".")

        # keyword highlighting
        def highlight_keywords(txt: str) -> str:
            kws = ["urgent","verify","verification","confirm","suspend","deactivate","reset",
                   "password","unusual activity","account","terminate","refund","invoice","payment"]
            out = txt
            for kw in sorted(kws, key=len, reverse=True):
                out = re.sub(rf"(?i)({re.escape(kw)})", r"<mark>\1</mark>", out)
            return out
        with st.expander("Preview with keyword highlights"):
            st.markdown(f"<div style='white-space:pre-wrap'>{highlight_keywords(raw_text)}</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error("❌ Inference failed. Ensure the model is the HYBRID pipeline.")
        st.exception(e)

# ─────────────────────────────────────────────────────────────────────────────
# Batch mode
with st.expander("🧪 Batch test (multiple emails)"):
    multi = st.text_area("Separate emails with a line containing only ---", height=200, key="batch_in")
    if st.button("Run batch"):
        chunks = [c.strip() for c in (multi or "").split("\n---\n") if c.strip()]
        if chunks:
            probs = model.predict_proba(chunks)[:, 1]
            preds = (probs >= threshold).astype(int)
            df = pd.DataFrame({
                "email": chunks,
                "prob_phish": probs,
                "label": ["PHISHING" if p else "LEGIT" for p in preds]
            })
            st.dataframe(df, use_container_width=True)
            st.download_button("Download CSV", df.to_csv(index=False).encode(), "batch_results.csv", "text/csv")
        else:
            st.warning("No emails found.")

st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("🔧 Model: TF-IDF + URL/Email Features + Logistic Regression • Built with Streamlit")
