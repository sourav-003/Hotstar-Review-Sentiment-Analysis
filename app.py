import os
import re
import string
import joblib
import numpy as np
import gradio as gr
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import tensorflow as tf

# ---- NLTK setup ----
def _ensure_nltk():
    import nltk
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)

    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)

# Call the setup function here
_ensure_nltk()

stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)
wd = WordNetLemmatizer()

# ---- Text preprocessing (matched to the notebook) ----
def preprocess_text(text: str) -> str:
    pattern = r"""@[a-zA-Z0-9_:]+|b['"]rt|[\d]+[a-zA-Z_+='?]+[\d]+[\d]+|[a-zA-Z_*+=]+[\d]+[a-zA-Z_*+-=]+|[\d]+"""
    pattern = pattern + r"""|https:+[a-zA-Z0-9/._+-=]+|&amp;|rt"""
    review = re.sub(pattern, "", text)
    stop_free = " ".join(t for t in review.lower().split() if t not in stop_words)
    puct_free = " ".join(t for t in stop_free.split() if t not in punctuation)
    final_words = puct_free.replace("#", "")
    clean_review = " ".join(wd.lemmatize(word) for word in final_words.split())
    return clean_review

# ---- Load fitted artifacts (created by the notebook) ----
VEC_PATH = os.getenv("TFIDF_PATH", "tfidf_vectorizer.pkl")
SVD_PATH = os.getenv("SVD_PATH", "svd_reducer.pkl")
MODEL_PATH = os.getenv("MODEL_PATH", "deep_learning_model.h5")

# Load lazily on first request to allow the app to start even if files are missing
_ARTIFACTS = {"loaded": False, "tfidf": None, "svd": None, "model": None}

def _load_artifacts():
    if _ARTIFACTS["loaded"]:
        return _ARTIFACTS["tfidf"], _ARTIFACTS["svd"], _ARTIFACTS["model"]
    try:
        tfidf_vectorizer: TfidfVectorizer = joblib.load(VEC_PATH)
        svd_reducer: TruncatedSVD = joblib.load(SVD_PATH)
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model artifacts. Ensure these files exist in the working directory: "
            f"{VEC_PATH}, {SVD_PATH}, {MODEL_PATH}. Original error: {e}"
        )
    _ARTIFACTS.update({"loaded": True, "tfidf": tfidf_vectorizer, "svd": svd_reducer, "model": model})
    return tfidf_vectorizer, svd_reducer, model

# Mapping used in the notebook discussion/comments:
# 0 -> Negative, 1 -> Neutral, 2 -> Positive
CLASS_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

def predict_sentiment(review: str) -> str:
    if not review or not review.strip():
        return "Please enter some text."
    tfidf_vectorizer, svd_reducer, model = _load_artifacts()
    cleaned = preprocess_text(review)
    X_tfidf = tfidf_vectorizer.transform([cleaned])
    X_svd = svd_reducer.transform(X_tfidf)
    probs = model.predict(X_svd, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = CLASS_MAP.get(idx, str(idx))
    return f"{label} (confidence={probs[idx]:.3f})"

DESCRIPTION = (
    "Enter a review to get its predicted sentiment (Positive, Negative, or Neutral). "
    "Model: TF-IDF → SVD → Dense NN (loaded from saved artifacts)."
)

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=5, label="Enter Review"),
    outputs=gr.Textbox(label="Predicted Sentiment"),
    title="Hotstar Review Sentiment Analysis (Deep Learning Model)",
    description=DESCRIPTION,
    examples=[
        ["The streaming quality is amazing and the content is top-notch!"],
        ["It keeps buffering and the app crashes too often."],
        ["It's okay, nothing special."],
    ],
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
