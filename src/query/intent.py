"""Query intent classification — routes queries to optimal search behavior.

Intent types:
- keyword:    Exact match needed (ISBN, specific title)
- semantic:   Abstract concept, needs vector search
- author:     Looking for books by specific author
- similar_to: "Books like X" → find X, use its embedding
- filtered:   Has explicit constraints (year, genre)

Lightweight rule-based approach — no ML model, <1ms.
"""

import re
from dataclasses import dataclass, field
from enum import Enum


class Intent(str, Enum):
    keyword = "keyword"
    semantic = "semantic"
    author = "author"
    similar_to = "similar_to"
    filtered = "filtered"


@dataclass
class QueryAnalysis:
    """Result of analyzing a user query."""
    original: str
    corrected: str
    was_corrected: bool
    intent: Intent
    search_mode: str  # hybrid, vector, keyword
    boost_keyword: float = 1.0  # weight for BM25 vs vector
    filters: dict = field(default_factory=dict)
    extracted_title: str = ""  # for similar_to intent
    confidence: float = 0.0


# Patterns for intent detection
SIMILAR_PATTERNS = [
    r"(?:books?|novels?|stories?)\s+(?:like|similar\s+to|resembling)\s+(.+)",
    r"(?:similar|like)\s+(.+)",
    r"(?:if\s+(?:i|you)\s+liked?)\s+(.+)",
    r"(?:recommend.*(?:based on|like))\s+(.+)",
]

AUTHOR_PATTERNS = [
    r"(?:books?\s+)?by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:books?|novels?|works?|writing)",
    r"(?:author|written by)\s+(.+)",
]

ISBN_PATTERN = r"(?:978|979)[\-\s]?\d[\-\s]?\d{2}[\-\s]?\d{6}[\-\s]?\d"

YEAR_PATTERN = r"(?:(?:published|from|after|before|in)\s+)?(\d{4})"
DECADE_PATTERN = r"(\d{4})s"

GENRE_KEYWORDS = {
    "romance", "mystery", "thriller", "horror", "fantasy", "sci-fi",
    "science fiction", "historical fiction", "memoir", "biography",
    "cookbook", "self-help", "poetry", "graphic novel", "young adult",
    "children", "literary fiction", "crime", "adventure", "western",
}

CONCEPT_SIGNALS = [
    "about", "exploring", "theme of", "meaning of", "stories of",
    "books about", "novels about", "dealing with", "understanding",
]


def classify_intent(query: str) -> QueryAnalysis:
    """Classify query intent and determine optimal search strategy."""
    query_lower = query.lower().strip()

    # --- ISBN lookup ---
    if re.search(ISBN_PATTERN, query):
        return QueryAnalysis(
            original=query, corrected=query, was_corrected=False,
            intent=Intent.keyword, search_mode="keyword",
            confidence=0.95,
        )

    # --- "Similar to" / "Books like X" ---
    for pattern in SIMILAR_PATTERNS:
        match = re.search(pattern, query_lower)
        if match:
            title = match.group(1).strip().strip('"\'')
            return QueryAnalysis(
                original=query, corrected=query, was_corrected=False,
                intent=Intent.similar_to, search_mode="vector",
                extracted_title=title, confidence=0.9,
            )

    # --- Author queries ---
    for pattern in AUTHOR_PATTERNS:
        match = re.search(pattern, query)
        if match:
            return QueryAnalysis(
                original=query, corrected=query, was_corrected=False,
                intent=Intent.author, search_mode="hybrid",
                boost_keyword=2.0, confidence=0.8,
            )

    # --- Extract year filters ---
    filters = {}
    year_match = re.search(YEAR_PATTERN, query_lower)
    decade_match = re.search(DECADE_PATTERN, query_lower)
    if year_match:
        year = int(year_match.group(1))
        if "after" in query_lower or "since" in query_lower:
            filters["year_min"] = year
        elif "before" in query_lower:
            filters["year_max"] = year
        else:
            filters["year_min"] = year
            filters["year_max"] = year
    elif decade_match:
        decade = int(decade_match.group(1))
        filters["year_min"] = decade
        filters["year_max"] = decade + 9

    # --- Concept/abstract queries → prefer vector ---
    is_concept = any(signal in query_lower for signal in CONCEPT_SIGNALS)
    if is_concept and not filters:
        return QueryAnalysis(
            original=query, corrected=query, was_corrected=False,
            intent=Intent.semantic, search_mode="hybrid",
            boost_keyword=0.5, confidence=0.7,
        )

    # --- Filtered queries ---
    if filters:
        return QueryAnalysis(
            original=query, corrected=query, was_corrected=False,
            intent=Intent.filtered, search_mode="hybrid",
            filters=filters, confidence=0.7,
        )

    # --- Default: hybrid (balanced) ---
    return QueryAnalysis(
        original=query, corrected=query, was_corrected=False,
        intent=Intent.semantic, search_mode="hybrid",
        confidence=0.5,
    )
