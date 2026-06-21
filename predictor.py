"""
predictor.py — Cyberbullying Detection Engine (Tier 1 → Tier 3)

Memuat semua model dari folder `models/` dan menyediakan fungsi predict
untuk setiap tier.
"""

import os, re, json, warnings
import joblib
import numpy as np
import torch
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings("ignore")

nltk.download("stopwords", quiet=True)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

STOP_EN = set(stopwords.words("english"))

def preprocess_classical(text: str) -> str:
    """Preprocessing untuk classical ML (TF-IDF): lowercase, hapus URL/mention/hashtag/digit/punc/stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in STOP_EN and len(t) > 1]
    result = " ".join(tokens)
    return result if result else text.lower().strip()


EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE,
)

def preprocess_bert(text: str) -> str:
    """Light preprocessing untuk BERT — preserve linguistic structure."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r'^["\u201C\u201D]+|["\u201C\u201D]+$', "", text)
    text = EMOJI_RE.sub(" ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else text.strip()


def _load_pkl(filename):
    path = os.path.join(MODEL_DIR, filename)
    return joblib.load(path)

def _load_json(filename):
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "r") as f:
        return json.load(f)


class CyberbullyingPredictor:
    """
    Memuat semua model dan menyediakan method predict_all(text)
    yang mengembalikan hasil prediksi Tier 1 s.d. Tier 3.
    """

    def __init__(self, enable_bert: bool = True):
        self.enable_bert = enable_bert
        self._load_label_info()
        self._load_classical_models()
        if enable_bert:
            self._load_bert_models()

    def _load_label_info(self):
        info = _load_json("label_classes.json")
        self.target_names = info["classes"]
        self.n_classes = info["n_classes"]

    def _load_classical_models(self):
        self.tfidf = _load_pkl("tfidf_vectorizer.pkl")

        self.t1_models = {
            "LR":  _load_pkl("t1_lr.pkl"),
            "NB":  _load_pkl("t1_nb.pkl"),
            "SVM": _load_pkl("t1_svm.pkl"),
            "RF":  _load_pkl("t1_rf.pkl"),
            "KNN": _load_pkl("t1_knn.pkl"),
        }

        self.wsv_weights = _load_pkl("wsv_weights.pkl")

        self.hard_voting = _load_pkl("t2b_hard_voting.pkl")

        self.stacking_2c = _load_pkl("stacking_2c.pkl")

        self.stacking_2d = _load_pkl("stacking_2d.pkl")

        self.best_classical = _load_pkl("best_classical_ml.pkl")
        name_path = os.path.join(MODEL_DIR, "best_classical_ml_name.txt")
        with open(name_path, "r") as f:
            self.best_classical_name = f.read().strip()

    def _load_bert_models(self):
        bert_dir = os.path.join(MODEL_DIR, "bert_3a_best")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(bert_dir)
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(bert_dir)
        self.bert_model.to(self.device)
        self.bert_model.eval()

        self.meta_lr = _load_pkl("meta_lr_3b.pkl")


    def _tfidf_transform(self, text: str):
        """Preprocess + TF-IDF transform satu teks."""
        clean = preprocess_classical(text)
        return self.tfidf.transform([clean])

    def predict_tier1(self, text: str) -> dict:
        """Tier 1: prediksi individual 5 model (LR, NB, SVM, RF, KNN)."""
        X = self._tfidf_transform(text)
        results = {}
        for name, model in self.t1_models.items():
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            results[name] = {
                "label": self.target_names[pred],
                "confidence": float(proba.max()),
                "probabilities": {
                    self.target_names[i]: float(p) for i, p in enumerate(proba)
                },
            }
        return results

    def predict_tier2a(self, text: str) -> dict:
        """Tier 2A: Weighted Soft Voting (bobot = val F1)."""
        X = self._tfidf_transform(text)
        names = ["LR", "NB", "SVM", "RF", "KNN"]
        w_arr = np.array([self.wsv_weights[n] for n in names])

        probs_all = np.stack(
            [self.t1_models[n].predict_proba(X)[0] for n in names], axis=0
        )
        p_wsv = (probs_all * w_arr.reshape(-1, 1)).sum(axis=0) / w_arr.sum()
        pred = p_wsv.argmax()

        return {
            "label": self.target_names[pred],
            "confidence": float(p_wsv[pred]),
            "probabilities": {
                self.target_names[i]: float(p) for i, p in enumerate(p_wsv)
            },
        }

    def predict_tier2b(self, text: str) -> dict:
        """Tier 2B: Hard Voting (DT + RF + XGB)."""
        X = self._tfidf_transform(text)
        pred = self.hard_voting.predict(X)[0]

        individual_preds = [
            est.predict(X)[0] for est in self.hard_voting.estimators_
        ]
        n_est = len(individual_preds)
        agreement = individual_preds.count(pred) / n_est

        vote_probs = {
            self.target_names[i]: individual_preds.count(i) / n_est
            for i in range(self.n_classes)
        }

        return {
            "label": self.target_names[pred],
            "confidence": agreement,
            "probabilities": vote_probs,
        }

    def predict_tier2c(self, text: str) -> dict:
        """Tier 2C: Stacking (LR + NB + SVM → LR)."""
        X = self._tfidf_transform(text)
        pred = self.stacking_2c.predict(X)[0]
        proba = self.stacking_2c.predict_proba(X)[0]
        return {
            "label": self.target_names[pred],
            "confidence": float(proba.max()),
            "probabilities": {
                self.target_names[i]: float(p) for i, p in enumerate(proba)
            },
        }

    def predict_tier2d(self, text: str) -> dict:
        """Tier 2D: Stacking (DT + RF + XGB → RF)."""
        X = self._tfidf_transform(text)
        pred = self.stacking_2d.predict(X)[0]
        proba = self.stacking_2d.predict_proba(X)[0]
        return {
            "label": self.target_names[pred],
            "confidence": float(proba.max()),
            "probabilities": {
                self.target_names[i]: float(p) for i, p in enumerate(proba)
            },
        }

    def _bert_predict_proba(self, text: str) -> np.ndarray:
        """Internal: jalankan BERT inference, return probability vector."""
        clean = preprocess_bert(text)
        inputs = self.tokenizer(
            clean,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.bert_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return probs

    def predict_tier3a(self, text: str) -> dict:
        """Tier 3A: BERT + Weighted Loss."""
        if not self.enable_bert:
            return {"error": "BERT tidak di-load. Enable di sidebar."}

        probs = self._bert_predict_proba(text)
        pred = probs.argmax()
        return {
            "label": self.target_names[pred],
            "confidence": float(probs[pred]),
            "probabilities": {
                self.target_names[i]: float(p) for i, p in enumerate(probs)
            },
        }

    def predict_tier3b(self, text: str) -> dict:
        """Tier 3B: Enhanced BERT (BERT probs + best classical probs → meta-LR)."""
        if not self.enable_bert:
            return {"error": "BERT tidak di-load. Enable di sidebar."}

        bert_probs = self._bert_predict_proba(text)

        X_tfidf = self._tfidf_transform(text)
        ml_probs = self.best_classical.predict_proba(X_tfidf)[0]

        meta_features = np.concatenate([bert_probs, ml_probs]).reshape(1, -1)
        pred = self.meta_lr.predict(meta_features)[0]
        proba = self.meta_lr.predict_proba(meta_features)[0]

        return {
            "label": self.target_names[pred],
            "confidence": float(proba.max()),
            "probabilities": {
                self.target_names[i]: float(p) for i, p in enumerate(proba)
            },
        }

    def predict_all(self, text: str) -> dict:
        """Jalankan semua tier sekaligus."""
        results = {
            "input_text": text,
            "tier1": self.predict_tier1(text),
            "tier2a_wsv": self.predict_tier2a(text),
            "tier2b_hard_voting": self.predict_tier2b(text),
            "tier2c_stacking": self.predict_tier2c(text),
            "tier2d_stacking": self.predict_tier2d(text),
        }
        if self.enable_bert:
            results["tier3a_bert"] = self.predict_tier3a(text)
            results["tier3b_enhanced"] = self.predict_tier3b(text)
        return results