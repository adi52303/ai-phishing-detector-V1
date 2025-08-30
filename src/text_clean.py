# src/text_clean.py
import re, string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure resources (safe if already present)
try:
    _ = stopwords.words("english")
except:
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("punkt_tab")  # some NLTK versions need this
    nltk.download("wordnet")

STOPWORDS = set(stopwords.words("english"))
LEMM = WordNetLemmatizer()

def simple_clean(text: str) -> str:
    if not isinstance(text, str): 
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)  # mask URLs
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [LEMM.lemmatize(t) for t in tokens if t not in STOPWORDS and t.isalpha()]
    return " ".join(tokens)
