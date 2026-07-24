"""
Text cleaning and normalization utilities for OpenLibrary data.
"""

import json
import re


def parse_description(val: object) -> str | None:
    """Parse description from OL's various formats (JSON-encoded or plain string)."""
    if val is None:
        return None
    if not isinstance(val, str) or not val.strip():
        return None

    text = val.strip()

    # OL stores descriptions as JSON: {"type":"/type/text","value":"..."}
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            text = parsed.get("value", "").strip()
        except (json.JSONDecodeError, AttributeError):
            pass

    if not text or len(text) < 15:
        return None

    # Filter out physical descriptions (page counts, dimensions)
    if re.match(r"^\d+\s*(p\.|pages|v\.|vol|leaves)", text, re.IGNORECASE):
        return None
    if re.match(r"^[\d\s,]+p\b", text):
        return None
    if re.match(r".*;\s*\d+\s*cm\.?$", text) and len(text) < 40:
        return None

    return text.strip()


def normalize_subjects(subjects: list[str] | None) -> list[str]:
    """Normalize and deduplicate subject tags."""
    if not subjects:
        return []

    seen = set()
    normalized = []
    for s in subjects:
        if not isinstance(s, str):
            continue
        clean = s.strip().lower()
        # Remove trailing periods
        clean = clean.rstrip(".")
        # Skip empty or very short
        if len(clean) < 2:
            continue
        # Skip if already seen (case-insensitive dedup)
        if clean in seen:
            continue
        seen.add(clean)
        # Store with title case for readability
        normalized.append(clean.title())

    return normalized[:15]  # cap at 15 subjects


def extract_year(date_str: str | None) -> int | None:
    """Extract a 4-digit year from OL's inconsistent date formats."""
    if not date_str or not isinstance(date_str, str):
        return None
    match = re.search(r"(\d{4})", date_str)
    if match:
        year = int(match.group(1))
        if 1000 <= year <= 2030:
            return year
    return None


def is_likely_english(text: str) -> bool:
    """Rough heuristic to detect English text."""
    en_words = {
        "the", "a", "an", "is", "are", "was", "were", "and", "of", "in",
        "to", "for", "this", "that", "with", "from", "by", "on", "it",
        "as", "at", "be", "or", "not", "but", "have", "has", "had",
        "her", "his", "who", "about", "into", "when", "which", "their",
    }
    words = set(text.lower().split()[:30])
    overlap = len(words & en_words)
    return overlap >= 2
