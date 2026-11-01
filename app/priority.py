"""
Rule-based ticket priority detection.

To add a new urgent indicator, just append a word or phrase to URGENT_KEYWORDS.
Phrases may contain spaces ("not working"); matching is case-insensitive and
only matches whole words, so "down" does NOT match "download".
"""

import re
from dataclasses import dataclass, field

# Edit this list to change what counts as urgent.
URGENT_KEYWORDS = [
    "urgent",
    "down",
    "not working",
    "critical",
    "immediately",
]


@dataclass
class PriorityResult:
    label: str                                   # "Urgent" or "Normal"
    matched_keywords: list = field(default_factory=list)

    @property
    def is_urgent(self) -> bool:
        return self.label == "Urgent"


def _keyword_pattern(keyword: str) -> re.Pattern:
    """Whole-word, case-insensitive pattern; spaces match any whitespace."""
    words = [re.escape(part) for part in keyword.strip().lower().split()]
    return re.compile(r"(?<!\w)" + r"\s+".join(words) + r"(?!\w)", flags=re.IGNORECASE)


def detect_priority(subject: str, body: str, keywords=None) -> PriorityResult:
    """Return Urgent if any keyword appears in the subject or body, else Normal."""
    keywords = URGENT_KEYWORDS if keywords is None else keywords
    text = f"{subject or ''} {body or ''}"
    matched = [kw for kw in keywords if kw.strip() and _keyword_pattern(kw).search(text)]
    return PriorityResult(label="Urgent" if matched else "Normal", matched_keywords=matched)
