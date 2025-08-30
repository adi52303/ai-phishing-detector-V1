# src/features.py
import re
import pandas as pd
from urllib.parse import urlparse


SAFE_DOMAINS = {
    "codingninjas.com", "youtube.com", "youtu.be", "linkedin.com",
    "instagram.com", "facebook.com", "iitg.ac.in", "iitguwahati.ac.in",
    "google.com", "drive.google.com", "zoom.us", "notion.so"
}

SHORTENERS = {
    "bit.ly","tinyurl.com","t.co","goo.gl","ow.ly","is.gd","buff.ly","cutt.ly","rebrand.ly"
}

# Trimmed suspicious words – avoid common marketing words to reduce false positives
SUSPICIOUS_WORDS = {
    # 🔒 Account / Security threats
    "password", "suspend", "limited", "security", "confirm", "login", "verify", 
    "reactivate", "unlock", "reset", "deactivate", "credentials", "unauthorized", 
    "blocked", "lock", "update account", "revalidate",

    # ⚠️ Urgency / Fear triggers
    "urgent", "immediately", "expire", "warning", "alert", "attention", 
    "action required", "final notice", "last chance", "limited time", 
    "important", "critical", "now", "instantly", "asap", "response needed",

    # 💳 Financial / Payment baits
    "bank", "account", "billing", "invoice", "payment", "transaction", 
    "refund", "tax", "irs", "lottery", "prize", "jackpot", "bonus", 
    "crypto", "bitcoin", "wire transfer", "western union", "winnings", 
    "payout", "compensation", "claim reward",

    # 🔗 Clickbait / Malicious links
    "click", "link", "here", "download", "attachment", "access now", 
    "open", "update", "login here", "tap", "visit", "url", "http", "https",
    "safe login", "confirm link", "secure portal",

    # 🏛️ Spoofed Authority / Brands
    "official", "secure", "trusted", "verify identity", "government", 
    "support", "admin", "compliance", "apple", "paypal", "google", 
    "microsoft", "outlook", "amazon", "netflix", "facebook", 
    "instagram", "bank of america", "hsbc", "citibank",

    # 🕵️ Social Engineering
    "urgent request", "confidential", "restricted", "identity theft", 
    "authentication", "credentials required", "keep safe", 
    "not shared", "important message", "verify ownership", 
    "account compromised", "unusual activity", "security check",

    # 🎁 Too Good To Be True
    "congratulations", "winner", "exclusive offer", "guaranteed", 
    "act fast", "no risk", "free trial", "free access", "zero cost", 
    "special promotion", "gift card", "voucher", "limited offer"
}

SUSPICIOUS_TLDS = {".ru", ".xyz", ".zip", ".top", ".quest", ".cam", ".click"}

URL_REGEX = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
IP_REGEX  = re.compile(r'https?://(\d{1,3}\.){3}\d{1,3}')

def _normalize_domain(u: str) -> str:
    try:
        u2 = u if u.lower().startswith(("http://","https://")) else "http://" + u
        netloc = urlparse(u2).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except:
        return ""

def extract_url_features(text: str) -> dict:
    if not isinstance(text, str):
        text = ""

    urls = URL_REGEX.findall(text)
    domains = []
    for u in urls:
        d = _normalize_domain(u)
        if d:
            domains.append(d)

    num_urls = len(urls)
    has_url = int(num_urls > 0)
    has_ip_url = int(bool(IP_REGEX.search(text)))
    has_shortener = int(any(d in SHORTENERS for d in domains))
    has_safe_domain = int(any(any(d == sd or d.endswith("." + sd) for sd in SAFE_DOMAINS) for d in domains))
    # flag suspicious TLDs and deep subdomains (e.g., a.b.c.example.com)
    tld_suspicious = int(any(any(d.endswith(tld) for tld in SUSPICIOUS_TLDS) for d in domains))
    max_dot_count = max((d.count(".") for d in domains), default=0)

    lower = text.lower()
    has_suspicious_word = int(any(w in lower for w in SUSPICIOUS_WORDS))

    return {
        "num_urls": num_urls,
        "has_url": has_url,
        "has_ip_url": has_ip_url,
        "has_shortener": has_shortener,
        "has_safe_domain": has_safe_domain,          # NEW: de-risks known brands
        "tld_suspicious": tld_suspicious,            # NEW
        "max_dot_count": max_dot_count,              # NEW
        "has_suspicious_word": has_suspicious_word,
        # For debugging in the app (optional; model ignores these next two)
        "_domains": ", ".join(domains[:8]),
        "_flags": ""
    }

def to_feature_frame(X):
    rows = [extract_url_features(t) for t in X]
    # Drop debug-only columns so the model sees only numeric features
    df = pd.DataFrame(rows)
    debug_cols = ["_domains", "_flags"]
    for c in debug_cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
    return df

