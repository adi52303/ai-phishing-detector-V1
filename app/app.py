# app/app.py
import os, sys
import joblib
import pandas as pd
import streamlit as st

# --------------------
# Fix Python path so we can import from src/
# --------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from features import to_feature_frame, extract_url_features  # now works

# --------------------
# Load model
# --------------------
MODEL_PATH = os.path.join(ROOT, "models", "tfidf_logreg.joblib")
model = joblib.load(MODEL_PATH)

# --------------------
# Streamlit Page Config
# --------------------
st.set_page_config(
    page_title="AI Phishing Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --------------------
# Header
# --------------------
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🛡️ AI Phishing Detector</h1>", unsafe_allow_html=True)
st.write("Paste an email (subject + body). The model predicts **Phishing** or **Legit** with probability and feature explanations.")

# --------------------
# Threshold slider
# --------------------
threshold = st.slider(
    "Decision threshold (lower = stricter, higher = lenient)",
    0.0, 1.0, 0.30, 0.01
)

# --------------------
# Input Box
# --------------------
text = st.text_area(
    "✉️ Email text",
    height=220,
    placeholder="Paste an email here..."
)

# --------------------
# Feature Debugger
# --------------------
def _debug_flags(text: str) -> dict:
    info = extract_url_features(text or "")
    return {
        "num_urls": info["num_urls"],
        "has_url": info["has_url"],
        "has_ip_url": info["has_ip_url"],
        "has_shortener": info["has_shortener"],
        "has_safe_domain": info["has_safe_domain"],
        "tld_suspicious": info["tld_suspicious"],
        "max_dot_count": info["max_dot_count"],
        "has_suspicious_word": info["has_suspicious_word"],
        "domains": info.get("_domains", "")
    }

# --------------------
# Main Button
# --------------------
if st.button("🔍 Analyze"):
    if not text.strip():
        st.warning("⚠️ Please paste some text.")
    else:
        # Prediction
        X = pd.DataFrame({"text": [text]})
        proba = float(model.predict_proba(X)[0][1])  # P(phishing)
        label = "PHISHING" if proba >= threshold else "LEGIT"

        # UI Styling
        if label == "PHISHING":
            st.markdown(
                f"<h2 style='color:#C0392B;'>🚨 Prediction: {label}</h2>", unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<h2 style='color:#27AE60;'>✅ Prediction: {label}</h2>", unsafe_allow_html=True
            )

        # Probability Meter
        st.progress(min(max(proba, 0.0), 1.0))
        st.metric("Phishing Probability", f"{proba:.3f}")

        st.caption("Tip: Adjust the threshold above to tune sensitivity.")

        # Explainability
        # Explainability (User-Friendly)
    with st.expander("📊 Why this decision? (feature signals)"):
        flags = _debug_flags(text)

        explanations = {
            "num_urls": f"🔗 URLs found: {flags['num_urls']}",
            "has_url": "🌐 Contains a link" if flags["has_url"] else "❌ No links detected",
            "has_ip_url": "⚠️ Contains an IP address in link" if flags["has_ip_url"] else "✅ No IP-based links",
            "has_shortener": "⚠️ Uses URL shortener (bit.ly, tinyurl)" if flags["has_shortener"] else "✅ No shorteners",
            "has_safe_domain": "✅ Known safe domain detected" if flags["has_safe_domain"] else "❌ No safe domain match",
            "tld_suspicious": "⚠️ Suspicious domain ending (.xyz, .top, .tk)" if flags["tld_suspicious"] else "✅ Normal domain endings",
            "max_dot_count": f"🔍 Long/suspicious domain depth: {flags['max_dot_count']}" if flags["max_dot_count"] > 2 else "✅ Domain structure looks normal",
            "has_suspicious_word": "⚠️ Contains phishing words (urgent, verify, password)" if flags["has_suspicious_word"] else "✅ No suspicious keywords",
            "domains": f"🌐 Domains found: {flags['domains']}" if flags["domains"] else "❌ No domains detected"
        }

        for key, explanation in explanations.items():
            st.write(explanation)


# --------------------
# Footer
# --------------------
st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("🔧 Model: TF-IDF + URL Features + Logistic Regression | Built by Aditya Das with ❤️ using Streamlit")
