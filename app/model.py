"""
Prediction layer: loads the saved model + vectorizer and classifies tickets.

ensure_artifacts() makes the project self-starting:
  * no data/tickets.csv      -> generate it
  * no saved model files     -> train them
  * saved with another scikit-learn version, or unreadable -> retrain
Training is skipped when valid files already exist, so the server starts fast.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import joblib
import sklearn

from app.generate_dataset import DATA_PATH, save_dataset
from app.preprocessing import preprocess_ticket
from app.priority import PriorityResult, detect_priority
from app.train_model import (CONFIDENCE_THRESHOLD, METRICS_PATH, MODEL_PATH,
                             VECTORIZER_PATH, train_and_save)

REVIEW_LABEL = "Needs Human Review"


@dataclass
class Prediction:
    department: str                 # model's top choice
    confidence: float               # 0..1, probability of the top choice
    probabilities: dict = field(default_factory=dict)  # department -> probability
    needs_review: bool = False
    priority: PriorityResult | None = None

    @property
    def display_department(self) -> str:
        return REVIEW_LABEL if self.needs_review else self.department


def _artifacts_are_valid() -> bool:
    if not (MODEL_PATH.exists() and VECTORIZER_PATH.exists() and METRICS_PATH.exists()):
        return False
    try:
        saved = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        return saved.get("sklearn_version") == sklearn.__version__
    except (OSError, ValueError):
        return False


def ensure_artifacts() -> None:
    """Create the dataset and/or trained model if they are missing."""
    if not DATA_PATH.exists():
        print("[TicketFlow] data/tickets.csv not found - generating dataset...")
        save_dataset()
    if not _artifacts_are_valid():
        print("[TicketFlow] Training model (first run or scikit-learn version changed)...")
        train_and_save()
        print("[TicketFlow] Training finished.")


class TicketClassifier:
    """Wraps the saved model and vectorizer."""

    def __init__(self, threshold: float = CONFIDENCE_THRESHOLD):
        ensure_artifacts()
        try:
            self.model = joblib.load(MODEL_PATH)
            self.vectorizer = joblib.load(VECTORIZER_PATH)
        except Exception:  # corrupted or incompatible pickle: rebuild once
            train_and_save()
            self.model = joblib.load(MODEL_PATH)
            self.vectorizer = joblib.load(VECTORIZER_PATH)
        self.threshold = threshold

    def load_metrics(self) -> dict:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    def predict(self, subject: str, body: str) -> Prediction:
        """Classify one ticket. Raises ValueError if there is no usable text."""
        cleaned = preprocess_ticket(subject, body)   # identical to training
        if not cleaned:
            raise ValueError("The ticket has no readable text after cleaning.")

        features = self.vectorizer.transform([cleaned])
        probabilities = self.model.predict_proba(features)[0]
        best = int(probabilities.argmax())
        confidence = float(probabilities[best])

        return Prediction(
            department=str(self.model.classes_[best]),
            confidence=confidence,
            probabilities={str(c): float(p) for c, p in zip(self.model.classes_, probabilities)},
            needs_review=confidence < self.threshold,
            priority=detect_priority(subject, body),
        )
