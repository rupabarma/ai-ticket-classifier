"""
Train and evaluate the ticket classifier.

    python -m app.train_model

Pipeline: preprocess -> TF-IDF (English stop words, unigrams + bigrams)
          -> Logistic Regression.
All numbers saved to models/metrics.json are computed from the held-out test
set. Nothing is typed in by hand.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

from app.preprocessing import preprocess_ticket

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "tickets.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "ticket_classifier.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

CONFIDENCE_THRESHOLD = 0.60
TEST_SIZE = 0.20
RANDOM_STATE = 42
MAX_ITER = 1000
C_VALUE = 10.0  # inverse regularisation strength; higher = more confident predictions


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    return df


def train_and_save(data_path: Path = DATA_PATH) -> dict:
    """Train, evaluate on the test split, save artifacts and metrics. Returns metrics."""
    df = load_dataset(data_path)
    texts = [preprocess_ticket(s, b) for s, b in zip(df["subject"], df["body"])]
    labels = df["department"].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )

    # Fit the vectorizer on the training split only, so no test text leaks in.
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = LogisticRegression(max_iter=MAX_ITER, C=C_VALUE)
    model.fit(x_train_vec, y_train)

    # ---- Evaluation on the held-out test set ----
    classes = list(model.classes_)
    y_pred = model.predict(x_test_vec)
    proba = model.predict_proba(x_test_vec)
    top_conf = proba.max(axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    w_p, w_r, w_f, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    per_class = classification_report(y_test, y_pred, labels=classes, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, labels=classes, digits=3, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred, labels=classes)

    # How the 60% rule behaves on the test set
    auto = top_conf >= CONFIDENCE_THRESHOLD
    auto_accuracy = float(accuracy_score(y_test[auto], y_pred[auto])) if auto.any() else None
    review_correct = float(np.mean(y_test[~auto] == y_pred[~auto])) if (~auto).any() else None

    metrics = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "sklearn_version": sklearn.__version__,
        "dataset_rows": int(len(df)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "model_params": {"algorithm": "LogisticRegression", "C": C_VALUE, "max_iter": MAX_ITER,
                         "ngram_range": [1, 2], "stop_words": "english", "test_fraction": TEST_SIZE,
                         "random_state": RANDOM_STATE},
        "accuracy": float(accuracy),
        "macro": {"precision": float(macro_p), "recall": float(macro_r), "f1": float(macro_f)},
        "weighted": {"precision": float(w_p), "recall": float(w_r), "f1": float(w_f)},
        "per_class": {c: {k: float(per_class[c][k]) for k in ("precision", "recall", "f1-score")}
                      | {"support": int(per_class[c]["support"])} for c in classes},
        "classes": classes,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report_text,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "threshold_check": {
            "auto_classified": int(auto.sum()),
            "needs_review": int((~auto).sum()),
            "accuracy_on_auto_classified": auto_accuracy,
            "accuracy_on_needs_review": review_correct,
            "mean_confidence": float(top_conf.mean()),
        },
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    result = train_and_save()
    print(f"Train: {result['train_size']}  Test: {result['test_size']}  Vocabulary: {result['vocabulary_size']}")
    print(f"Accuracy: {result['accuracy']:.4f}\n")
    print(result["classification_report"])
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(result["classes"])
    for row in result["confusion_matrix"]:
        print(row)
    print("\n60% rule on test set:", result["threshold_check"])
    print(f"\nSaved: {MODEL_PATH.name}, {VECTORIZER_PATH.name}, {METRICS_PATH.name}")
