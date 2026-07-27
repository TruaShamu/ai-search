"""Known-item evaluation dataset — sampled from the indexed corpus.

Generates a set of books whose titles are unique within the corpus
(no collisions), suitable for known-item retrieval probes.  The set is
persisted to ``data/eval/known_items.json`` so evaluations are stable
and reproducible across runs.

Also generates "hard variant" probes (partial titles, typos,
title+author) reported separately from the headline accuracy number.

Usage as library::

    from src.eval.known_items import load_known_items, load_hard_variants
    items = load_known_items()         # list[dict]
    hard  = load_hard_variants()       # list[dict]

Usage as CLI (regenerate from corpus)::

    python -m src.eval.known_items [--count 50] [--seed 42] [--corpus PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


# Paths relative to repo root (resolved at call time, never hardcoded absolute)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CORPUS = _REPO_ROOT / "data" / "processed" / "books_augmented.jsonl"
_OUTPUT_DIR = _REPO_ROOT / "data" / "eval"
_KNOWN_ITEMS_PATH = _OUTPUT_DIR / "known_items.json"
_HARD_VARIANTS_PATH = _OUTPUT_DIR / "known_item_hard_variants.json"

# Title-length heuristic: very short titles are rarely distinctive
_MIN_TITLE_WORDS = 3
_MIN_TITLE_CHARS = 10

# Words excluded from document-frequency analysis (function words)
_STOPWORDS = frozenset({
    "a", "an", "the", "of", "and", "or", "in", "on", "at", "to", "for",
    "is", "it", "by", "with", "from", "as", "be", "was", "are", "were",
    "been", "has", "had", "have", "do", "does", "did", "not", "no", "but",
    "if", "than", "that", "this", "its", "my", "his", "her", "our", "their",
    "he", "she", "we", "they", "me", "him", "us", "them", "who", "what",
    "so", "up", "out", "all", "about", "into", "over", "after", "before",
})

# Fraction of corpus a word must appear in to be considered "common"
# (computed over title + description text, not titles alone).
# 0.5% of 26K docs = ~133 documents.
_COMMON_WORD_DF_THRESHOLD = 0.005


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_known_items(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load the persisted known-item set (or regenerate if missing)."""
    p = Path(path) if path else _KNOWN_ITEMS_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Known-item set not found at {p}. "
            "Run `python -m src.eval.known_items` to generate it."
        )
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_hard_variants(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load the hard-variant probes (partial title, typo, title+author)."""
    p = Path(path) if path else _HARD_VARIANTS_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Hard-variant set not found at {p}. "
            "Run `python -m src.eval.known_items` to generate it."
        )
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _load_indexed_books(corpus_path: Path) -> list[dict[str, Any]]:
    """Load books that have a non-empty description (= indexed in Qdrant)."""
    books: list[dict[str, Any]] = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("description"):
                books.append(obj)
    return books


def _find_unique_titles(books: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter to books whose title is unique (case-insensitive) in the corpus."""
    title_counts: Counter[str] = Counter()
    for b in books:
        title_counts[b["title"].strip().lower()] += 1

    unique = []
    for b in books:
        key = b["title"].strip().lower()
        if title_counts[key] == 1:
            unique.append(b)
    return unique


def _is_distinctive(title: str) -> bool:
    """Heuristic: a title is distinctive if it is long-ish and not generic."""
    words = title.split()
    if len(words) < _MIN_TITLE_WORDS or len(title) < _MIN_TITLE_CHARS:
        return False
    # Reject titles that are all-caps (often acronyms/series labels)
    if title == title.upper() and len(title) > 5:
        return False
    return True


def _tokenize_title(text: str) -> list[str]:
    """Lowercase split, strip punctuation, remove stopwords."""
    return [
        w for w in re.sub(r"[^\w\s]", "", text.lower()).split()
        if w not in _STOPWORDS and len(w) > 1
    ]


def compute_corpus_word_df(
    books: list[dict[str, Any]],
) -> tuple[Counter[str], int]:
    """Compute document frequency over indexed text (title + description).

    This reflects what TF-IDF actually sees -- words that are common across
    descriptions have low IDF and give the lexical retriever little signal.

    Returns (word_df_counter, total_document_count).
    """
    df: Counter[str] = Counter()
    for b in books:
        text = b.get("title", "") + " " + (b.get("description", "") or "")
        words = set(_tokenize_title(text))
        for w in words:
            df[w] += 1
    return df, len(books)


def classify_title(
    title: str,
    word_df: Counter[str],
    total_docs: int,
    threshold: float = _COMMON_WORD_DF_THRESHOLD,
) -> str:
    """Classify a title as ``distinctive`` or ``common_words``.

    A title is ``common_words`` if ALL its content words appear in more than
    *threshold* fraction of corpus titles.  Otherwise it is ``distinctive``
    (at least one word is rare enough to give TF-IDF discriminating signal).
    """
    words = _tokenize_title(title)
    if not words:
        return "common_words"
    for w in words:
        if word_df.get(w, 0) / total_docs < threshold:
            return "distinctive"
    return "common_words"


def _make_typo(title: str, rng: random.Random) -> str:
    """Introduce a single realistic typo into a title."""
    words = title.split()
    if len(words) < 2:
        return title
    # Pick a word with >3 chars to mutate
    candidates = [i for i, w in enumerate(words) if len(w) > 3]
    if not candidates:
        return title
    idx = rng.choice(candidates)
    word = words[idx]
    # Swap two adjacent characters
    pos = rng.randint(1, len(word) - 2)
    mutated = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
    words[idx] = mutated
    return " ".join(words)


def _make_partial(title: str, rng: random.Random) -> str:
    """Extract a meaningful partial title (first N words or drop subtitle)."""
    # If title has a colon or dash, take the part before it
    for sep in [":", " - ", " — "]:
        if sep in title:
            parts = title.split(sep)
            return parts[0].strip()
    # Otherwise take first ~60% of words
    words = title.split()
    n = max(2, int(len(words) * 0.6))
    return " ".join(words[:n])


def _generate_hard_variants(
    items: list[dict[str, Any]], rng: random.Random, count: int = 10
) -> list[dict[str, Any]]:
    """Generate deliberately hard known-item variants for diagnostic probing."""
    variants: list[dict[str, Any]] = []
    sample = rng.sample(items, min(count, len(items)))

    for item in sample:
        work_id = item["work_id"]
        title = item["title"]
        authors = item.get("authors", [])
        author_str = authors[0] if authors else ""

        # Partial title
        variants.append({
            "query": _make_partial(title, rng),
            "work_id": work_id,
            "title": title,
            "variant_type": "partial_title",
        })

        # Typo in title
        variants.append({
            "query": _make_typo(title, rng),
            "work_id": work_id,
            "title": title,
            "variant_type": "typo",
        })

        # Title + author
        if author_str:
            variants.append({
                "query": f"{title} {author_str}",
                "work_id": work_id,
                "title": title,
                "variant_type": "title_plus_author",
            })

    return variants


def generate_known_items(
    corpus_path: Path = _DEFAULT_CORPUS,
    count: int = 50,
    seed: int = 42,
    hard_variant_count: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Sample a known-item set from the indexed corpus.

    Returns (known_items, hard_variants).
    """
    rng = random.Random(seed)

    print(f"Loading indexed books from {corpus_path} ...")
    books = _load_indexed_books(corpus_path)
    print(f"  {len(books)} indexed books loaded")

    unique = _find_unique_titles(books)
    print(f"  {len(unique)} have corpus-unique titles")

    distinctive = [b for b in unique if _is_distinctive(b["title"])]
    print(f"  {len(distinctive)} pass distinctiveness filter")

    if len(distinctive) < count:
        print(f"  WARNING: only {len(distinctive)} distinctive titles, "
              f"requested {count}")
        count = len(distinctive)

    sampled = rng.sample(distinctive, count)

    known_items = []
    for b in sampled:
        wid = b["work_id"].replace("/works/", "")
        known_items.append({
            "query": b["title"],
            "work_id": wid,
            "title": b["title"],
            "authors": b.get("authors", []),
        })

    # Classify titles by corpus word frequency (DF over descriptions)
    word_df, total = compute_corpus_word_df(books)
    n_distinctive = 0
    n_common = 0
    for item in known_items:
        cls = classify_title(item["title"], word_df, total)
        item["title_word_class"] = cls
        if cls == "distinctive":
            n_distinctive += 1
        else:
            n_common += 1
    print(f"  Title-word classification: {n_distinctive} distinctive, "
          f"{n_common} common-word titles")

    hard_variants = _generate_hard_variants(
        known_items, rng, count=hard_variant_count
    )

    return known_items, hard_variants


def save_known_items(
    items: list[dict[str, Any]],
    hard_variants: list[dict[str, Any]],
    output_dir: Path = _OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Persist known-item and hard-variant sets to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    items_path = output_dir / "known_items.json"
    variants_path = output_dir / "known_item_hard_variants.json"

    with open(items_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    with open(variants_path, "w", encoding="utf-8") as f:
        json.dump(hard_variants, f, indent=2, ensure_ascii=False)

    return items_path, variants_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate known-item evaluation set from indexed corpus."
    )
    parser.add_argument(
        "--corpus", type=Path, default=_DEFAULT_CORPUS,
        help="Path to books_augmented.jsonl",
    )
    parser.add_argument("--count", type=int, default=50,
                        help="Number of known items to sample (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducibility (default: 42)")
    parser.add_argument("--hard-variants", type=int, default=10,
                        help="Number of books to generate hard variants for")
    parser.add_argument("--output-dir", type=Path, default=_OUTPUT_DIR,
                        help="Output directory for JSON files")
    args = parser.parse_args()

    items, hard_variants = generate_known_items(
        corpus_path=args.corpus,
        count=args.count,
        seed=args.seed,
        hard_variant_count=args.hard_variants,
    )

    items_path, variants_path = save_known_items(
        items, hard_variants, output_dir=args.output_dir
    )

    print(f"\nSaved {len(items)} known items to {items_path}")
    print(f"Saved {len(hard_variants)} hard variants to {variants_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
