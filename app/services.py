"""
Application logic behind the pages (no web-framework code in here).

main.py stays thin: it receives requests and hands them to TicketService,
which returns plain dictionaries that the Jinja2 templates render.
"""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime

import pandas as pd

from app.generate_dataset import DATA_PATH
from app.model import REVIEW_LABEL, TicketClassifier
from app.priority import URGENT_KEYWORDS

MAX_SUBJECT_LEN = 200
MAX_BODY_LEN = 5000
RECENT_LIMIT = 8


class TicketService:
    def __init__(self):
        self.classifier = TicketClassifier()
        self.metrics = self.classifier.load_metrics()
        self.dataset_stats = self._load_dataset_stats()
        self._lock = threading.Lock()
        self._recent = deque(maxlen=RECENT_LIMIT)
        self._session = {"total": 0, "auto": 0, "review": 0, "urgent": 0}

    # ---------- dataset ----------
    @staticmethod
    def _load_dataset_stats() -> dict:
        df = pd.read_csv(DATA_PATH, encoding="utf-8")
        counts = df["department"].value_counts()
        total = int(len(df))
        biggest = int(counts.max())
        distribution = [
            {"department": str(name), "count": int(n), "share": round(100 * n / total, 1),
             "bar": round(100 * n / biggest, 1)}
            for name, n in sorted(counts.items())
        ]
        return {"total": total, "departments": len(counts), "distribution": distribution}

    # ---------- page contexts ----------
    def dashboard_context(self) -> dict:
        with self._lock:
            recent = list(self._recent)
            session = dict(self._session)
        return {
            "active": "dashboard",
            "total_tickets": self.dataset_stats["total"],
            "department_count": self.dataset_stats["departments"],
            "accuracy_pct": round(self.metrics["accuracy"] * 100, 1),
            "session": session,
            "distribution": self.dataset_stats["distribution"],
            "recent": recent,
            "threshold_pct": int(self.classifier.threshold * 100),
        }

    def classify_page_context(self, subject="", body="", errors=None, result=None) -> dict:
        return {
            "active": "classify",
            "subject": subject,
            "body": body,
            "errors": errors or {},
            "result": result,
            "threshold_pct": int(self.classifier.threshold * 100),
            "max_subject": MAX_SUBJECT_LEN,
            "max_body": MAX_BODY_LEN,
        }

    def metrics_context(self) -> dict:
        m = self.metrics
        classes = m["classes"]
        matrix = m["confusion_matrix"]
        rows = []
        for actual, counts in zip(classes, matrix):
            row_total = sum(counts) or 1
            cells = []
            for predicted, value in zip(classes, counts):
                share = value / row_total
                cells.append({"value": value, "alpha": round(0.08 + 0.92 * share, 2),
                              "dark": share >= 0.5, "diagonal": actual == predicted})
            rows.append({"actual": actual, "cells": cells, "total": row_total})
        per_class = [{"department": c, **m["per_class"][c]} for c in classes]
        return {
            "active": "metrics",
            "m": m,
            "accuracy_pct": round(m["accuracy"] * 100, 1),
            "classes": classes,
            "matrix_rows": rows,
            "per_class": per_class,
            "threshold_pct": int(m["confidence_threshold"] * 100),
        }

    def about_context(self) -> dict:
        return {
            "active": "about",
            "threshold_pct": int(self.classifier.threshold * 100),
            "keywords": URGENT_KEYWORDS,
            "dataset_total": self.dataset_stats["total"],
            "sklearn_version": self.metrics.get("sklearn_version", ""),
        }

    # ---------- classification ----------
    def classify_submission(self, subject: str, body: str) -> dict:
        """Validate the form input, classify, and return the classify-page context."""
        subject = (subject or "").strip()
        body = (body or "").strip()

        errors = {}
        if not subject:
            errors["subject"] = "Enter a subject so the ticket has a title."
        elif len(subject) > MAX_SUBJECT_LEN:
            errors["subject"] = f"Subject is too long. Keep it under {MAX_SUBJECT_LEN} characters."
        if not body:
            errors["body"] = "Describe the issue so the model has something to read."
        elif len(body) > MAX_BODY_LEN:
            errors["body"] = f"Description is too long. Keep it under {MAX_BODY_LEN} characters."
        if errors:
            return self.classify_page_context(subject, body, errors=errors)

        try:
            prediction = self.classifier.predict(subject, body)
        except ValueError:
            errors["body"] = "No readable words found. Use letters or numbers in the description."
            return self.classify_page_context(subject, body, errors=errors)
        except Exception:  # unexpected failure: show a friendly message, keep the app alive
            errors["general"] = "Something went wrong while classifying this ticket. Please try again."
            return self.classify_page_context(subject, body, errors=errors)

        result = self._result_dict(prediction)
        self._record(subject, result)
        return self.classify_page_context(subject, body, result=result)

    def _result_dict(self, p) -> dict:
        ranked = sorted(p.probabilities.items(), key=lambda kv: kv[1], reverse=True)
        return {
            "department": p.display_department,
            "top_department": p.department,
            "needs_review": p.needs_review,
            "confidence_pct": round(p.confidence * 100, 1),
            "status": "Needs human review" if p.needs_review else "Auto-classified",
            "priority": p.priority.label,
            "is_urgent": p.priority.is_urgent,
            "matched_keywords": p.priority.matched_keywords,
            "probabilities": [{"department": d, "pct": round(v * 100, 1), "top": d == p.department}
                              for d, v in ranked],
            "review_label": REVIEW_LABEL,
        }

    def _record(self, subject: str, result: dict) -> None:
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "subject": subject if len(subject) <= 60 else subject[:57] + "...",
            "department": result["department"],
            "top_department": result["top_department"],
            "confidence_pct": result["confidence_pct"],
            "priority": result["priority"],
            "needs_review": result["needs_review"],
        }
        with self._lock:
            self._recent.appendleft(entry)
            self._session["total"] += 1
            self._session["review" if result["needs_review"] else "auto"] += 1
            if result["is_urgent"]:
                self._session["urgent"] += 1
