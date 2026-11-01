"""
Text preprocessing shared by training AND prediction.

Keeping this in ONE place guarantees that a ticket is cleaned the same way
when the model learns from it and when a user submits it on the website.
"""

import re

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
SPECIAL_CHARS_PATTERN = re.compile(r"[^a-z0-9\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, drop URLs, drop special characters, collapse whitespace."""
    if text is None:
        return ""
    text = str(text).lower()
    text = URL_PATTERN.sub(" ", text)
    text = SPECIAL_CHARS_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def combine_fields(subject: str, body: str) -> str:
    """Join subject and body into one string (the subject comes first)."""
    return f"{subject or ''} {body or ''}".strip()


def preprocess_ticket(subject: str, body: str) -> str:
    """Full pipeline: combine subject + body, then clean the result."""
    return clean_text(combine_fields(subject, body))
