# src/features.py
# Strong, wide URL + text signal engineering for phishing detection.
# Pure stdlib + pandas (no external deps).

import re
import math
import unicodedata
import pandas as pd
from urllib.parse import urlparse

# -----------------------------
# Curated lists
# -----------------------------

# Common safe roots (expand as needed; include your org domain)
SAFE_DOMAINS = {
    # Your org / internal — add yours here:
    # "yourcompany.com", "corp.yourcompany.com",
    "company.com",  # placeholder example

    # Collaboration / meetings
    "zoom.us", "teams.microsoft.com", "meet.google.com", "calendar.google.com",
    "slack.com", "webex.com", "gotomeeting.com", "whereby.com",
    # Big clouds / dev
    "google.com", "docs.google.com", "drive.google.com", "mail.google.com",
    "dropbox.com", "box.com", "microsoft.com", "office.com", "live.com",
    "sharepoint.com", "github.com", "gitlab.com", "bitbucket.org",
    "aws.amazon.com", "azure.com", "cloudflare.com",
    # Productivity
    "notion.so", "figma.com", "canva.com", "airtable.com", "calendly.com",
    "atlassian.com", "jira.atlassian.com", "confluence.atlassian.com",
    "zoom.com",
    # Social/consumer
    "linkedin.com", "twitter.com", "x.com", "youtube.com", "youtu.be",
    "instagram.com", "facebook.com", "whatsapp.com", "spotify.com",
    # EDU / org examples
    "iitg.ac.in", "iitguwahati.ac.in",
    # Payments
    "stripe.com", "paypal.com", "pay.google.com", "squareup.com",
    # Commerce
    "amazon.com", "ebay.com", "shopify.com",
}

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "cutt.ly", "rebrand.ly", "lnkd.in", "shorturl.at", "rb.gy", "s.id",
    "v.gd", "shrtco.de", "bit.do", "t.ly"
}

SUSPICIOUS_TLDS = {
    ".ru", ".cn", ".tk", ".top", ".xyz", ".zip", ".mov", ".click", ".quest",
    ".cam", ".work", ".monster", ".country", ".gq", ".cf", ".ml", ".men",
    ".lol", ".win", ".info", ".biz", ".rest", ".support", ".review"
}

KW_URGENCY = {
    "urgent", "immediately", "asap", "action required", "final notice",
    "expires", "expire", "last chance", "within 24 hours"
}
KW_SECURITY = {
    "verify", "verification", "confirm", "suspend", "suspension", "locked",
    "deactivate", "reset", "security alert", "unusual activity", "compromised",
    "authentication", "2fa", "otp", "credentials"
}
KW_FINANCIAL = {
    "invoice", "payment", "billing", "refund", "tax", "bank", "wire transfer",
    "bitcoin", "crypto", "prize", "lottery", "jackpot", "gift card"
}
KW_ATTACHMENT = {
    "attachment", "attached", "pdf attached", "doc attached", "download file"
}

SECOND_LEVEL_SUFFIXES = {
    "co.uk", "ac.uk", "gov.uk", "org.uk",
    "com.au", "net.au", "org.au",
    "co.in", "com.br", "com.mx", "com.tr", "com.cn", "co.jp",
}

# -----------------------------
# Regexes
# -----------------------------
URL_REGEX = re.compile(r"""(?ix)
    (?:href\s*=\s*['"](?P<href>https?://[^'"]+)['"])  # HTML href
  | (?:\[(?P<md_text>[^\]]+)\]\((?P<md_url>https?://[^)]+)\)) # Markdown [text](url)
  | (?P<bare>https?://[^\s<>'")]+)                    # Bare URL
  | (?P<www>www\.[^\s<>'")]+)                         # Bare www.
""")
IP_REGEX     = re.compile(r"https?://(\d{1,3}\.){3}\d{1,3}", re.IGNORECASE)
EMAIL_REGEX  = re.compile(r'(?i)\b[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,})\b')

# -----------------------------
# Helpers
# -----------------------------
def _normalize_domain(u: str) -> str:
    try:
        u2 = u if u.lower().startswith(("http://","https://")) else "http://" + u
        netloc = urlparse(u2).netloc.lower().strip()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""

def _root_domain(domain: str) -> str:
    if not domain:
        return ""
    parts = domain.split(".")
    if len(parts) <= 2:
        return domain
    last2 = ".".join(parts[-2:])
    last3 = ".".join(parts[-3:])
    if last2 in SECOND_LEVEL_SUFFIXES and len(parts) >= 3:
        return ".".join(parts[-3:])
    if last3 in SECOND_LEVEL_SUFFIXES and len(parts) >= 4:
        return ".".join(parts[-4:])
    return ".".join(parts[-2:])

def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    from collections import Counter
    counts = Counter(s)
    n = float(len(s))
    return -sum((c/n) * math.log2(c/n) for c in counts.values())

def _pct_upper(s: str) -> float:
    if not s:
        return 0.0
    ups = sum(1 for ch in s if ch.isupper())
    return ups / max(1, len(s))

def _pct_non_ascii(s: str) -> float:
    if not s:
        return 0.0
    non = sum(1 for ch in s if ord(ch) > 127)
    return non / max(1, len(s))

def _looks_confusable(s: str) -> int:
    for ch in s:
        try:
            name = unicodedata.name(ch)
            if "CYRILLIC" in name or "GREEK" in name:
                return 1
        except Exception:
            pass
    return int(any(ord(ch) > 127 for ch in s))

def _domain_tld(domain: str) -> str:
    if not domain or "." not in domain:
        return ""
    return "." + domain.split(".")[-1]

def _extract_urls(text: str):
    urls = []
    for m in URL_REGEX.finditer(text or ""):
        href = m.group("href")
        if href:
            urls.append(("", href)); continue
        md_text, md_url = m.group("md_text"), m.group("md_url")
        if md_url:
            urls.append((md_text or "", md_url)); continue
        bare = m.group("bare")
        if bare:
            urls.append(("", bare)); continue
        www = m.group("www")
        if www:
            urls.append(("", "http://" + www)); continue
    return urls

def _display_vs_href_mismatch(display: str, href: str) -> int:
    disp_dom = _normalize_domain(display) if display else ""
    href_dom = _normalize_domain(href)
    if not disp_dom:
        m = re.search(r"\b([a-z0-9.-]+\.[a-z]{2,})\b", display or "", flags=re.I)
        if m:
            disp_dom = _normalize_domain(m.group(1))
    if disp_dom and href_dom and _root_domain(disp_dom) != _root_domain(href_dom):
        return 1
    return 0

# -----------------------------
# Core feature extraction
# -----------------------------
def extract_url_features(text: str) -> dict:
    if not isinstance(text, str):
        text = ""

    # URLs
    links = _extract_urls(text)
    url_domains, url_root_domains, url_lengths, domain_entropies = [], [], [], []
    mismatch_flag, at_in_path = 0, 0

    for display, href in links:
        dom = _normalize_domain(href)
        rd  = _root_domain(dom)
        if dom: url_domains.append(dom)
        if rd:  url_root_domains.append(rd)
        if href: url_lengths.append(len(href))
        if dom: domain_entropies.append(_shannon_entropy(dom))
        mismatch_flag |= _display_vs_href_mismatch(display, href)
        try:
            path = urlparse(href).path or ""
            if "@" in path: at_in_path = 1
        except Exception:
            pass

    num_urls = len(links)
    has_url = int(num_urls > 0)
    has_ip_url = int(bool(IP_REGEX.search(text)))

    # Email addresses (From/To/Reply-To present in pasted text)
    email_domains = [m.group(1).lower() for m in EMAIL_REGEX.finditer(text or "")]
    email_root_domains = [_root_domain(d) for d in email_domains]
    num_emails = len(email_domains)
    has_email = int(num_emails > 0)

    # Combine URL + Email domains
    all_domains = set(url_domains) | set(email_domains)
    all_roots   = set(url_root_domains) | set(email_root_domains)

    num_domains = len(all_domains)
    num_root_domains = len(all_roots)
    num_shorteners = sum(1 for d in all_domains if d in SHORTENERS)
    num_suspicious_tlds = sum(1 for d in all_domains if _domain_tld(d) in SUSPICIOUS_TLDS)
    tld_suspicious = int(num_suspicious_tlds > 0)

    def _is_safe_root(rd: str) -> bool:
        return (rd in SAFE_DOMAINS) or any(rd.endswith("." + sd) for sd in SAFE_DOMAINS)
    num_safe_domains = sum(1 for rd in all_roots if _is_safe_root(rd))

    max_dot_count = max((d.count(".") for d in all_domains), default=0)
    avg_url_len = (sum(url_lengths) / num_urls) if num_urls else 0.0
    max_url_len = max(url_lengths) if num_urls else 0
    avg_dom_entropy = (sum(domain_entropies) / len(domain_entropies)) if domain_entropies else 0.0
    max_dom_entropy = max(domain_entropies) if domain_entropies else 0.0

    lower = (text or "").lower()
    kw_urg = sum(1 for w in KW_URGENCY if w in lower)
    kw_sec = sum(1 for w in KW_SECURITY if w in lower)
    kw_fin = sum(1 for w in KW_FINANCIAL if w in lower)
    kw_att = sum(1 for w in KW_ATTACHMENT if w in lower)
    kw_total = kw_urg + kw_sec + kw_fin + kw_att

    num_exclam = lower.count("!")
    num_digits = sum(ch.isdigit() for ch in text)
    pct_upper = _pct_upper(text)
    pct_non_ascii = _pct_non_ascii(text)
    has_confusable = _looks_confusable(text)

    return {
        "num_urls": num_urls,
        "num_emails": num_emails,
        "num_domains": num_domains,
        "num_root_domains": num_root_domains,
        "has_url": has_url,
        "has_email": has_email,
        "has_ip_url": has_ip_url,
        "num_shorteners": num_shorteners,
        "num_safe_domains": num_safe_domains,
        "num_suspicious_tlds": num_suspicious_tlds,
        "tld_suspicious": tld_suspicious,
        "max_dot_count": max_dot_count,
        "avg_url_len": float(avg_url_len),
        "max_url_len": int(max_url_len),
        "avg_domain_entropy": float(avg_dom_entropy),
        "max_domain_entropy": float(max_dom_entropy),
        "at_in_path": int(at_in_path > 0),
        "mismatch_display_href": mismatch_flag,
        "kw_urgency": kw_urg,
        "kw_security": kw_sec,
        "kw_financial": kw_fin,
        "kw_attachment": kw_att,
        "kw_total": kw_total,
        "num_exclamations": num_exclam,
        "num_digits": num_digits,
        "pct_uppercase": float(pct_upper),
        "pct_non_ascii": float(pct_non_ascii),
        "has_confusable_chars": has_confusable,
    }

FEATURE_COLUMNS = [
    "num_urls", "num_emails",
    "num_domains", "num_root_domains",
    "has_url", "has_email",
    "has_ip_url",
    "num_shorteners", "num_safe_domains", "num_suspicious_tlds",
    "tld_suspicious", "max_dot_count", "avg_url_len", "max_url_len",
    "avg_domain_entropy", "max_domain_entropy", "at_in_path",
    "mismatch_display_href",
    "kw_urgency", "kw_security", "kw_financial", "kw_attachment", "kw_total",
    "num_exclamations", "num_digits", "pct_uppercase", "pct_non_ascii",
    "has_confusable_chars",
]

def to_feature_frame(X):
    if isinstance(X, pd.DataFrame):
        if "text" in X.columns:
            iterable = X["text"].astype(str).tolist()
        else:
            iterable = X.astype(str).agg(" ".join, axis=1).tolist()
    elif isinstance(X, pd.Series):
        iterable = X.astype(str).tolist()
    elif isinstance(X, (list, tuple)):
        iterable = [str(t) for t in X]
    else:
        iterable = [str(X)]

    rows = [extract_url_features(t) for t in iterable]
    df = pd.DataFrame(rows)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    for c in list(df.columns):
        if c not in FEATURE_COLUMNS:
            df.drop(columns=[c], inplace=True)

    df = df[FEATURE_COLUMNS].copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df
