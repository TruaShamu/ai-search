"""Shared passage-building logic for reranker backends."""

import re

from src.reranker.config import MAX_DESCRIPTION_CHARS

_MACHINE_METADATA_RE = re.compile(r"^\w+:\S+=|=\d{4}\b")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_subjects(raw: list[str], max_subjects: int = 5) -> list[str]:
    """Clean and deduplicate subject tags for natural prose rendering.

    Splits comma-separated entries, drops machine-metadata tokens,
    deduplicates case-insensitively, and removes strict substrings.
    """
    flat = []
    for entry in raw:
        flat.extend(part.strip() for part in entry.split(",") if part.strip())

    filtered = [s for s in flat if not _MACHINE_METADATA_RE.search(s)]

    seen: set[str] = set()
    deduped: list[str] = []
    for s in filtered:
        key = s.lower().replace("-", " ")
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    result = []
    for s in deduped:
        s_lower = s.lower()
        if not any(s_lower != o.lower() and s_lower in o.lower() for o in deduped):
            result.append(s)

    return result[:max_subjects]


def build_passage(doc: dict) -> str:
    """Build passage text from document as natural prose for cross-encoder.

    ms-marco was trained on web prose, so we avoid pipe-delimited metadata.
    The tokenizer's MAX_SEQUENCE_TOKENS limit handles final truncation;
    MAX_DESCRIPTION_CHARS is a cheap guard sized to keep the token limit as
    the binding constraint.  These two constants are coupled — see config.py.
    """
    parts = []
    title = doc.get("title", "")
    authors = doc.get("authors", "")
    if title and authors:
        parts.append(f"{title} by {authors}.")
    elif title:
        parts.append(f"{title}.")

    if doc.get("description"):
        desc = doc["description"].strip("\"'")
        desc = _WHITESPACE_RE.sub(" ", desc).strip()
        if desc:
            parts.append(desc[:MAX_DESCRIPTION_CHARS])

    if doc.get("subjects"):
        subjects = doc["subjects"]
        if isinstance(subjects, list):
            cleaned = clean_subjects(subjects)
            if cleaned:
                parts.append(f"This book covers {', '.join(cleaned)}.")

    return " ".join(parts)
