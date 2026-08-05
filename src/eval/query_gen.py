"""Corpus-grounded query generation for book search evaluation.

Samples real books from the indexed corpus, then generates search queries
anchored to those books so at least one gold-relevant document is guaranteed
to exist.  Eliminates the two flaws of the prior approach:
  (a) no more "orphan" queries with zero relevant docs in the catalog
  (b) the generator sees actual book text, breaking the correlated blind-spot
      that arises when the same model invents both queries and judgments.

Metric trustworthiness by category
-----------------------------------
  title_lookup  -- gold_is_complete=True.  MRR, Recall@k, NDCG all meaningful
                   as absolute numbers.
  author        -- gold_is_complete=True (all indexed works by the queried
                   author are included).  MRR and Recall@k are real measures
                   of author-search quality.
  genre_topic   -- gold_is_complete=False.  Only the seed book is gold-labeled;
                   many other corpus books may legitimately match.  Recall@k
                   will be understated.  MRR is meaningful for *relative*
                   comparison between retrieval strategies (same gold set for
                   all), but the absolute number is not interpretable.
  exploratory   -- gold_is_complete=False (same caveat as genre_topic).
  combined      -- gold_is_complete=False (same caveat as genre_topic).

When reporting aggregate metrics, prefer MRR and per-category breakdowns.
Do NOT quote a single recall number across all categories as if it were a
true system recall -- the incomplete gold sets for topic/exploratory/combined
make that number meaninglessly low.

Usage:
    python -m src.eval.query_gen                        # default 100 queries, API
    python -m src.eval.query_gen --dry-run              # offline, template-based
    python -m src.eval.query_gen --n 50 --seed 42 --out data/eval/v2/queries_grounded.json
    python -m src.eval.query_gen --max-verbatim-ngram 3 # reject high-leakage queries
"""

from __future__ import annotations

import argparse
import json
import random
import re
import textwrap
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from src.eval.llm_client import AzureOpenAIClient, QUERY_GEN_RETRY_BACKOFF, QUERY_GEN_TIMEOUT

# ── Paths ────────────────────────────────────────────────────────────────────

# Corpus the queries are grounded in. This MUST match the corpus that is
# actually indexed and served: generated queries carry gold doc_ids drawn from
# this file, so pointing it at a stale corpus does not fail loudly -- it emits
# a query set whose gold ids resolve to nothing, and the eval reports a
# catastrophic-looking regression that is really just a wiring error.
# (The v1 file books_augmented.jsonl is still on disk, so this is a live trap.)
DEFAULT_CORPUS_PATH = Path("data/processed/books_goodreads_v2.jsonl")
DEFAULT_OUT = Path("data/eval/v2/queries_grounded.json")

# ── Category targets (match existing pipeline vocabulary) ────────────────────

CATEGORY_WEIGHTS: dict[str, float] = {
    "title_lookup": 0.20,
    "author": 0.15,
    "genre_topic": 0.30,
    "exploratory": 0.25,
    "combined": 0.10,
}

# Target difficulty distribution (30 / 40 / 30).
# Generation is tuned toward this, but it is a best-effort target; the
# achieved split is always reported honestly.
DIFFICULTY_TARGETS: dict[str, float] = {
    "easy": 0.30,
    "medium": 0.40,
    "hard": 0.30,
}

# ── Description-length buckets for stratification ───────────────────────────

_LEN_BINS = [
    ("short", 0, 174),       # ≤ p25
    ("medium", 175, 887),    # p25–p75
    ("long", 888, 999_999),  # > p75
]

# ── Broad genre mapping (top subjects → genre buckets) ──────────────────────

_GENRE_MAP: dict[str, str] = {
    "fiction": "fiction",
    "romance": "romance",
    "mystery": "mystery",
    "thriller": "thriller",
    "history": "history",
    "biography": "biography",
    "science": "science",
    "poetry": "poetry",
    "children": "children",
    "juvenile": "children",
    "philosophy": "philosophy",
    "religion": "religion",
    "art": "art",
    "music": "music",
    "education": "education",
    "politics": "politics",
    "economics": "economics",
    "travel": "travel",
    "animals": "nature",
    "nature": "nature",
    "comic": "comics",
    "fantasy": "fantasy",
    "horror": "horror",
    "cooking": "cooking",
    "health": "health",
    "technology": "technology",
    "computer": "technology",
    "mathematics": "science",
    "psychology": "psychology",
    "sociology": "social_science",
    "law": "law",
    "business": "business",
    "sports": "sports",
}

# ── Stopwords (small set, no NLTK dependency) ───────────────────────────────

STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "was", "are", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "not",
    "no", "nor", "so", "if", "then", "than", "that", "this", "these",
    "those", "i", "me", "my", "we", "our", "you", "your", "he", "him",
    "his", "she", "her", "they", "them", "their", "its", "what", "which",
    "who", "whom", "how", "when", "where", "why", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "only",
    "own", "same", "about", "up", "out", "just", "also", "very",
    "s", "t", "d", "m", "re", "ll", "ve", "book", "books", "novel",
    "novels", "story", "stories",
})

# ── Tokenization ────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, return token list."""
    return _TOKEN_RE.findall(text.lower())


def _tokenize_no_stop(text: str) -> list[str]:
    """Tokenize and remove stopwords."""
    return [t for t in _tokenize(text) if t not in STOPWORDS]

# ── Corpus loading ──────────────────────────────────────────────────────────

def load_indexed_books(path: Path = DEFAULT_CORPUS_PATH) -> list[dict]:
    """Load books that have a non-empty description (the indexed subset)."""
    books: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            book = json.loads(line)
            if book.get("description"):
                books.append(book)
    return books


def _book_searchable_text(book: dict) -> str:
    """Concatenate all searchable fields into one string."""
    parts = [
        book.get("title", ""),
        " ".join(book.get("authors", [])),
        " ".join(book.get("subjects", [])),
        book.get("description", "") or "",
    ]
    return " ".join(parts)


def build_corpus_df(books: list[dict]) -> dict[str, int]:
    """Build document-frequency index: term -> number of docs containing it."""
    df: dict[str, int] = defaultdict(int)
    for book in books:
        terms = set(_tokenize(_book_searchable_text(book)))
        for t in terms:
            df[t] += 1
    return dict(df)


def _normalize_author(name: str) -> str:
    """Conservative author normalization for matching."""
    n = name.lower().strip()
    n = re.sub(r"\s+", " ", n)
    for prefix in ("dr. ", "dr ", "prof. ", "prof ", "sir ", "lady ", "rev. ", "rev "):
        if n.startswith(prefix):
            n = n[len(prefix):]
    return n.strip()


def build_author_index(books: list[dict]) -> dict[str, list[str]]:
    """Build normalized-author -> list of work_ids for the indexed corpus.

    Uses conservative normalization (lowercase + strip titles) to avoid
    merging distinct people.
    """
    index: dict[str, list[str]] = defaultdict(list)
    for book in books:
        for author in book.get("authors", []):
            norm = _normalize_author(author)
            if norm:
                index[norm].append(book["work_id"])
    return dict(index)


def _broad_genre(book: dict) -> str:
    """Map a book's subjects to a single broad genre bucket."""
    subjects_lower = " ".join(s.lower() for s in book.get("subjects", []))
    for keyword, genre in _GENRE_MAP.items():
        if keyword in subjects_lower:
            return genre
    return "other"


def _desc_len_bucket(book: dict) -> str:
    desc_len = len(book.get("description", ""))
    for label, lo, hi in _LEN_BINS:
        if lo <= desc_len <= hi:
            return label
    return "long"


def _year_bucket(book: dict) -> str:
    year = book.get("first_publish_year")
    if not year:
        return "unknown"
    if year < 1900:
        return "pre-1900"
    if year < 1970:
        return "1900-1969"
    if year < 2000:
        return "1970-1999"
    return "2000+"


# ── Stratified seed sampling ───────────────────────────────────────────────

def _build_strata(books: list[dict]) -> dict[str, list[dict]]:
    """Group books into (genre, desc_len, year) strata."""
    strata: dict[str, list[dict]] = defaultdict(list)
    for b in books:
        key = f"{_broad_genre(b)}|{_desc_len_bucket(b)}|{_year_bucket(b)}"
        strata[key].append(b)
    return dict(strata)


def sample_seed_books(
    books: list[dict],
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Sample *n* seed books with stratification across genre, desc-length, and year."""
    strata = _build_strata(books)

    # Round-robin across strata so no single bucket dominates
    strata_keys = sorted(strata.keys())
    rng.shuffle(strata_keys)

    # Shuffle within each stratum
    for key in strata_keys:
        rng.shuffle(strata[key])

    seeds: list[dict] = []
    used_work_ids: set[str] = set()
    idx_per_stratum: dict[str, int] = defaultdict(int)

    round_robin_pos = 0
    while len(seeds) < n:
        key = strata_keys[round_robin_pos % len(strata_keys)]
        pool = strata[key]
        idx = idx_per_stratum[key]
        if idx < len(pool):
            book = pool[idx]
            idx_per_stratum[key] = idx + 1
            if book["work_id"] not in used_work_ids:
                used_work_ids.add(book["work_id"])
                seeds.append(book)
        round_robin_pos += 1
        # Safety: if we cycled through all strata without finding new books
        if round_robin_pos > len(strata_keys) * (n + 100):
            break

    # If we still need more (unlikely), fill randomly
    if len(seeds) < n:
        remaining = [b for b in books if b["work_id"] not in used_work_ids]
        rng.shuffle(remaining)
        seeds.extend(remaining[: n - len(seeds)])

    return seeds[:n]


# ── Assign categories to seed books ─────────────────────────────────────────

def assign_categories(n: int, rng: random.Random) -> list[str]:
    """Produce a category list of length *n* matching target distribution."""
    cats: list[str] = []
    for cat, weight in CATEGORY_WEIGHTS.items():
        count = round(n * weight)
        cats.extend([cat] * count)
    # Fix rounding: trim or pad to exactly n
    while len(cats) < n:
        cats.append(rng.choice(list(CATEGORY_WEIGHTS.keys())))
    cats = cats[:n]
    rng.shuffle(cats)
    return cats


# ── Dry-run (template) query generation ─────────────────────────────────────

def _add_typo(title: str, rng: random.Random) -> str:
    """Introduce a single realistic typo: swap two adjacent chars."""
    if len(title) < 4:
        return title
    # Pick a position in the middle (not first/last)
    pos = rng.randint(1, len(title) - 2)
    chars = list(title)
    chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    return "".join(chars)


def _partial_title(title: str, rng: random.Random) -> str:
    """Return a partial title: first N words or drop subtitle."""
    if ":" in title:
        return title.split(":")[0].strip()
    words = title.split()
    if len(words) <= 2:
        return title
    keep = rng.randint(max(1, len(words) // 2), len(words) - 1)
    return " ".join(words[:keep])


def generate_dry_run_query(
    book: dict,
    category: str,
    rng: random.Random,
) -> str:
    """Generate a template-based query with no API call.

    Tuned to produce a spread of difficulties (~30/40/30 easy/medium/hard).
    """
    title = book.get("title", "")
    authors = book.get("authors", [])
    subjects = book.get("subjects", [])
    description = book.get("description", "")
    year = book.get("first_publish_year")

    if category == "title_lookup":
        # Bias away from exact copies: exact=1, partial=3, typo=3
        variant = rng.choices(
            ["exact", "partial", "typo"], weights=[1, 3, 3], k=1
        )[0]
        if variant == "exact":
            return title
        elif variant == "partial":
            return _partial_title(title, rng)
        else:
            return _add_typo(title, rng)

    elif category == "author":
        if authors:
            author = rng.choice(authors)
            # Mix full-name (easy) and last-name-only (medium) variants
            variant = rng.choices(
                ["full", "last_name", "phrased"], weights=[2, 3, 3], k=1
            )[0]
            if variant == "full":
                return rng.choice([f"books by {author}", author])
            elif variant == "last_name":
                parts = author.split()
                last = parts[-1] if parts else author
                return rng.choice([
                    f"books by {last}",
                    f"{last} novels",
                    f"{last} author",
                ])
            else:
                return rng.choice([
                    f"{author} novels",
                    f"works of {author}",
                ])
        return f"books by unknown author about {subjects[0] if subjects else 'fiction'}"

    elif category == "genre_topic":
        if subjects:
            # Use only 1 subject for more medium/hard difficulty
            picked = rng.choice(subjects)
            templates = [
                f"{picked.lower()} books",
                f"books about {picked.lower()}",
                f"{picked.lower()}",
            ]
            return rng.choice(templates)
        return "general fiction books"

    elif category == "exploratory":
        # Pick a non-first sentence to reduce verbatim leakage
        sentences = [s.strip() for s in description.split(".") if len(s.strip()) > 20]
        if len(sentences) >= 2:
            sent = rng.choice(sentences[1:])
        elif sentences:
            sent = sentences[0]
        else:
            sent = description[:100]
        if len(sent) > 100:
            sent = sent[:100].rsplit(" ", 1)[0]
        templates = [
            f"book about {sent.lower()}",
            f"novel where {sent.lower()}",
            f"story involving {sent.lower()}",
            f"looking for {sent.lower()}",
        ]
        return rng.choice(templates)

    elif category == "combined":
        topic = subjects[0].lower() if subjects else "fiction"
        constraints = []
        if year:
            constraints.append(f"published around {year}")
        if len(subjects) > 1:
            constraints.append(f"related to {subjects[1].lower()}")
        if authors:
            constraints.append(f"similar to {authors[0]}")
        constraint = constraints[0] if constraints else "for adults"
        templates = [
            f"{topic} {constraint}",
            f"{constraint} {topic}",
            f"{topic} books {constraint}",
        ]
        return rng.choice(templates)

    return title  # fallback


# ── Azure OpenAI query generation ───────────────────────────────────────────

_GENERATION_PROMPT = textwrap.dedent("""\
You are helping build a search-engine evaluation benchmark for a book catalog.
Given a seed book's metadata, generate ONE realistic search query that a real
user might type to find this book or books like it.

SEED BOOK:
  Title: {title}
  Author(s): {authors}
  Subjects: {subjects}
  Year: {year}
  Description (first 500 chars): {description}

QUERY CATEGORY: {category}

Category instructions:
- title_lookup: A query based on the title. Make it realistic — use a partial
  title, a slight misspelling, or drop the subtitle. Do NOT just copy the
  exact title verbatim.
- author: A query focused on finding this author's works. Vary phrasing
  (e.g. "books by X", just the author name, "X novels").
- genre_topic: A topical/genre query using the book's real subject areas.
  Do NOT mention the title or author. Use 1-3 subject keywords naturally.
- exploratory: A natural-language, thematic, or conceptual query inspired
  by the description's actual themes. Should be multi-word and feel like
  a real user question. Do NOT mention the title or author.
- combined: A topic plus a constraint (era, genre, audience, etc.) that
  this seed book genuinely satisfies. Do NOT mention the title or author.

IMPORTANT:
- Return ONLY the query text, nothing else.
- Make it sound like something a real person would type into a search box.
- Vary difficulty: some should be easy (close to metadata), some harder
  (conceptual, oblique, or combining multiple facets).
- Do NOT lift distinctive phrases verbatim from the description.
""")


def _call_azure_openai(
    client: AzureOpenAIClient,
    url: str,
    headers: dict[str, str],
    prompt: str,
    temperature: float = 0.9,
) -> str | None:
    """Single Azure OpenAI call with retry."""
    del url, headers
    return client.call(
        user=prompt,
        temperature=temperature,
        max_tokens=150,
        strip_quotes=True,
    )


def generate_api_query(
    client: AzureOpenAIClient,
    url: str,
    headers: dict[str, str],
    book: dict,
    category: str,
) -> str | None:
    """Generate a single query via Azure OpenAI, grounded in a real book."""
    prompt = _GENERATION_PROMPT.format(
        title=book.get("title", ""),
        authors=", ".join(book.get("authors", [])),
        subjects=", ".join(book.get("subjects", [])[:8]),
        year=book.get("first_publish_year") or "unknown",
        description=(book.get("description", "") or "")[:500],
        category=category,
    )
    return _call_azure_openai(client, url, headers, prompt)


# ── Near-duplicate detection ────────────────────────────────────────────────

def _normalize_for_dedup(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def is_near_duplicate(
    query: str,
    existing: list[str],
    threshold: float = 0.75,
) -> bool:
    """Check if a query is too similar to any already-generated query."""
    norm = _normalize_for_dedup(query)
    for other in existing:
        ratio = SequenceMatcher(None, norm, _normalize_for_dedup(other)).ratio()
        if ratio >= threshold:
            return True
    return False


# ── Lexical-overlap leakage measurement ─────────────────────────────────────

def _max_verbatim_ngram_len(query_tokens: list[str], doc_tokens: list[str]) -> int:
    """Length of the longest contiguous token sequence present in both."""
    if not query_tokens or not doc_tokens:
        return 0
    max_possible = min(len(query_tokens), len(doc_tokens))
    for n in range(max_possible, 0, -1):
        doc_ngrams: set[tuple[str, ...]] = set()
        for j in range(len(doc_tokens) - n + 1):
            doc_ngrams.add(tuple(doc_tokens[j : j + n]))
        for i in range(len(query_tokens) - n + 1):
            if tuple(query_tokens[i : i + n]) in doc_ngrams:
                return n
    return 0


def lexical_overlap(
    query: str,
    book: dict,
    corpus_df: dict[str, int],
    corpus_size: int,
) -> dict[str, float]:
    """Measure lexical leakage between a query and its gold book.

    Returns dict with:
      unigram_jaccard      -- Jaccard similarity of non-stopword unigram sets
      max_verbatim_ngram   -- length of longest shared contiguous token run
      rare_term_overlap    -- fraction of query terms that are corpus-rare
                              (DF < 0.1% of corpus) AND appear in the gold doc
    """
    doc_text = _book_searchable_text(book)
    q_tokens = _tokenize_no_stop(query)
    d_tokens = _tokenize_no_stop(doc_text)

    q_set = set(q_tokens)
    d_set = set(d_tokens)

    # Unigram Jaccard
    if q_set or d_set:
        jaccard = len(q_set & d_set) / len(q_set | d_set)
    else:
        jaccard = 0.0

    # Max verbatim n-gram (on raw tokens including stopwords, for realism)
    q_raw = _tokenize(query)
    d_raw = _tokenize(doc_text)
    max_ngram = _max_verbatim_ngram_len(q_raw, d_raw)

    # Rare-term overlap
    rare_threshold = max(1, int(corpus_size * 0.001))
    if q_set:
        rare_and_in_doc = sum(
            1 for t in q_set
            if corpus_df.get(t, 0) < rare_threshold and t in d_set
        )
        rare_overlap = rare_and_in_doc / len(q_set)
    else:
        rare_overlap = 0.0

    return {
        "unigram_jaccard": round(jaccard, 4),
        "max_verbatim_ngram": max_ngram,
        "rare_term_overlap": round(rare_overlap, 4),
    }


# ── Difficulty annotation ───────────────────────────────────────────────────

def estimate_difficulty(query: str, book: dict, category: str) -> str:
    """Heuristic difficulty estimate: easy / medium / hard.

    Tuned so that the natural mix across categories lands closer to 30/40/30
    than the original 60/17/23.
    """
    title_lower = book.get("title", "").lower()
    query_lower = query.lower()

    if category == "title_lookup":
        ratio = SequenceMatcher(None, query_lower, title_lower).ratio()
        if ratio > 0.95:
            return "easy"
        elif ratio > 0.55:
            return "medium"
        return "hard"

    if category == "author":
        # Full author name present -> easy; partial -> medium; absent -> hard
        authors = book.get("authors", [])
        if any(a.lower() in query_lower for a in authors):
            return "easy"
        last_names = [a.split()[-1].lower() for a in authors if a.split()]
        if any(ln in query_lower for ln in last_names):
            return "medium"
        return "hard"

    if category == "exploratory":
        return "hard"

    if category == "combined":
        subjects_lower = {s.lower() for s in book.get("subjects", [])}
        sub_overlap = sum(1 for s in subjects_lower if s in query_lower)
        has_year = str(book.get("first_publish_year", "")) in query_lower
        if sub_overlap >= 2 or (sub_overlap >= 1 and has_year):
            return "easy"
        elif sub_overlap >= 1 or has_year:
            return "medium"
        return "hard"

    # genre_topic
    subjects_lower = {s.lower() for s in book.get("subjects", [])}
    overlap = sum(1 for s in subjects_lower if s in query_lower)
    if overlap >= 2:
        return "easy"
    elif overlap >= 1:
        return "medium"
    return "hard"


# ── Gold completeness ───────────────────────────────────────────────────────

def gold_is_complete(category: str) -> bool:
    """Whether the gold_work_ids list is a complete relevance set.

    True for title_lookup (one correct book) and author (all books by that
    author in the corpus).  False for topic/exploratory/combined where the
    corpus almost certainly contains additional relevant books beyond the
    single seed.
    """
    return category in ("title_lookup", "author")


# ── Main generation pipeline ───────────────────────────────────────────────

def generate_query_set(
    books: list[dict],
    n: int = 100,
    seed: int = 42,
    dry_run: bool = False,
    max_verbatim_ngram: int | None = None,
    corpus_df: dict[str, int] | None = None,
    author_index: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Generate *n* corpus-grounded queries.

    Returns a list of query objects compatible with the eval pipeline.
    """
    rng = random.Random(seed)

    # Build corpus infrastructure if not provided
    if corpus_df is None:
        corpus_df = build_corpus_df(books)
    if author_index is None:
        author_index = build_author_index(books)
    corpus_size = len(books)

    # Oversample seeds (2x) for dedup + leakage-filter headroom
    oversample_factor = 2.0
    n_seeds = min(int(n * oversample_factor), len(books))
    seed_books = sample_seed_books(books, n_seeds, rng)
    categories = assign_categories(n_seeds, rng)

    # Azure OpenAI setup (only if not dry-run)
    client: AzureOpenAIClient | None = None
    api_url = None
    api_headers = None
    if not dry_run:
        try:
            client = AzureOpenAIClient(
                timeout=QUERY_GEN_TIMEOUT,
                retry_backoff=QUERY_GEN_RETRY_BACKOFF,
            )
        except ValueError:
            print("WARNING: Azure OpenAI credentials not found. Falling back to dry-run mode.")
            dry_run = True

    queries: list[dict] = []
    existing_query_texts: list[str] = []
    used_work_ids_per_category: dict[str, set[str]] = defaultdict(set)
    leakage_rejected = 0
    api_failures = 0

    max_retries_per_slot = 3

    for i, (book, category) in enumerate(zip(seed_books, categories)):
        if len(queries) >= n:
            break

        # Avoid reusing the same work_id within a category
        if book["work_id"] in used_work_ids_per_category[category]:
            continue

        query_text = None
        leakage = None
        candidate_source = "template" if dry_run else "api"
        for retry in range(max_retries_per_slot):
            if dry_run:
                candidate = generate_dry_run_query(book, category, rng)
                candidate_source = "template"
            else:
                candidate = generate_api_query(
                    client, api_url, api_headers, book, category
                )
                if candidate is None:
                    api_failures += 1
                    candidate_source = "template_fallback"
                    candidate = generate_dry_run_query(book, category, rng)
                else:
                    candidate_source = "api"

            if not candidate or len(candidate.strip()) < 2:
                continue

            candidate = candidate.strip()

            # Near-duplicate check
            if is_near_duplicate(candidate, existing_query_texts):
                continue

            # Compute leakage
            leakage = lexical_overlap(candidate, book, corpus_df, corpus_size)

            # Optional leakage filter
            if (max_verbatim_ngram is not None
                    and leakage["max_verbatim_ngram"] >= max_verbatim_ngram):
                leakage_rejected += 1
                continue

            query_text = candidate
            break  # Good query found

        if query_text is None:
            continue  # Exhausted retries for this slot

        difficulty = estimate_difficulty(query_text, book, category)

        # -- Build gold_work_ids (complete author sets) --
        if category == "author":
            gold_ids: list[str] = []
            seen: set[str] = set()
            for author in book.get("authors", []):
                norm = _normalize_author(author)
                for wid in author_index.get(norm, []):
                    if wid not in seen:
                        seen.add(wid)
                        gold_ids.append(wid)
            # Safety cap: if normalization merged too many, fall back to seed
            if len(gold_ids) > 50:
                gold_ids = [book["work_id"]]
            elif not gold_ids:
                gold_ids = [book["work_id"]]
        else:
            gold_ids = [book["work_id"]]

        seed_book_record = {
            "work_id": book["work_id"],
            "title": book.get("title", ""),
            "authors": book.get("authors", []),
            "subjects": book.get("subjects", [])[:8],
            "year": book.get("first_publish_year"),
            "description_preview": (book.get("description", "") or "")[:200],
        }

        query_obj = {
            "query": query_text,
            "category": category,
            "gold_work_ids": gold_ids,
            "gold_is_complete": gold_is_complete(category),
            "difficulty": difficulty,
            "leakage": leakage,
            "source": candidate_source,
            "seed_book": seed_book_record,
        }

        queries.append(query_obj)
        existing_query_texts.append(query_text)
        used_work_ids_per_category[category].add(book["work_id"])

        if (len(queries)) % 20 == 0:
            print(f"  Generated {len(queries)}/{n} queries...")

    if client:
        client.close()

    if leakage_rejected:
        print(f"  Rejected {leakage_rejected} queries for max_verbatim_ngram >= {max_verbatim_ngram}")

    if api_failures:
        n_fallback = sum(1 for q in queries if q.get("source") == "template_fallback")
        print(f"  WARNING: {api_failures} API call(s) failed; "
              f"{n_fallback} kept query/queries came from template fallback")

    return queries


# ── Stats + output ──────────────────────────────────────────────────────────

def print_stats(queries: list[dict]) -> None:
    """Print category distribution, difficulty, and leakage diagnostics."""
    if not queries:
        print("No queries generated.")
        return

    cat_counts = Counter(q["category"] for q in queries)
    diff_counts = Counter(q["difficulty"] for q in queries)
    unique_gold = {wid for q in queries for wid in q["gold_work_ids"]}
    n = len(queries)

    print(f"\n{'='*60}")
    print(f"Generated {n} queries")
    print(f"{'='*60}")

    # -- Category distribution --
    print("\nCategory distribution:")
    for cat in CATEGORY_WEIGHTS:
        count = cat_counts.get(cat, 0)
        pct = 100 * count / n
        target_pct = CATEGORY_WEIGHTS[cat] * 100
        print(f"  {cat:<16} {count:>4}  ({pct:5.1f}%  target {target_pct:.0f}%)")

    # -- Difficulty distribution --
    print("\nDifficulty distribution (target 30/40/30):")
    for diff in ["easy", "medium", "hard"]:
        count = diff_counts.get(diff, 0)
        pct = 100 * count / n
        target_pct = DIFFICULTY_TARGETS[diff] * 100
        print(f"  {diff:<8} {count:>4}  ({pct:5.1f}%  target {target_pct:.0f}%)")

    # -- Gold completeness --
    complete = sum(1 for q in queries if q.get("gold_is_complete", False))
    incomplete = n - complete
    print(f"\nGold completeness: {complete} complete, {incomplete} incomplete")
    print("  (MRR reliable for all; Recall@k meaningful only for complete-gold queries)")

    # -- Author gold-set sizes --
    author_qs = [q for q in queries if q["category"] == "author"]
    if author_qs:
        gold_sizes = [len(q["gold_work_ids"]) for q in author_qs]
        print(f"\nAuthor gold-set sizes: min={min(gold_sizes)}, "
              f"median={sorted(gold_sizes)[len(gold_sizes)//2]}, "
              f"max={max(gold_sizes)}, mean={sum(gold_sizes)/len(gold_sizes):.1f}")

    print(f"\nUnique gold work_ids: {len(unique_gold)}")
    avg_query_len = sum(len(q["query"]) for q in queries) / n
    print(f"Mean query length:    {avg_query_len:.1f} chars")

    # -- Leakage diagnostics --
    leakage_present = [q for q in queries if "leakage" in q]
    if leakage_present:
        jaccards = sorted(q["leakage"]["unigram_jaccard"] for q in leakage_present)
        ngrams = sorted(q["leakage"]["max_verbatim_ngram"] for q in leakage_present)
        rares = sorted(q["leakage"]["rare_term_overlap"] for q in leakage_present)
        m = len(leakage_present)

        def _percentile(vals: list, p: float) -> float:
            idx = min(int(len(vals) * p), len(vals) - 1)
            return vals[idx]

        print(f"\nLeakage diagnostics ({m} queries):")
        print(f"  unigram_jaccard     median={_percentile(jaccards, 0.5):.4f}  "
              f"p90={_percentile(jaccards, 0.9):.4f}")
        print(f"  max_verbatim_ngram  median={_percentile(ngrams, 0.5):.0f}      "
              f"p90={_percentile(ngrams, 0.9):.0f}")
        print(f"  rare_term_overlap   median={_percentile(rares, 0.5):.4f}  "
              f"p90={_percentile(rares, 0.9):.4f}")

        high_leakage = sum(1 for v in ngrams if v >= 4)
        print(f"  Queries with max_verbatim_ngram >= 4 ('copied' threshold): "
              f"{high_leakage}/{m}")


def validate_gold_ids(queries: list[dict], books: list[dict]) -> bool:
    """Verify every gold_work_id exists in the indexed corpus."""
    indexed_ids = {b["work_id"] for b in books}
    all_ok = True
    for q in queries:
        for wid in q["gold_work_ids"]:
            if wid not in indexed_ids:
                print(f"  FAIL: gold_work_id {wid} not in indexed corpus "
                      f"(query: {q['query']!r})")
                all_ok = False
    return all_ok


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate corpus-grounded evaluation queries for book search."
    )
    parser.add_argument(
        "--n", type=int, default=100,
        help="Number of queries to generate (default: 100)",
    )
    parser.add_argument(
        "--out", type=str, default=str(DEFAULT_OUT),
        help=f"Output path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use template-based generation (no API calls)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--corpus", type=str, default=str(DEFAULT_CORPUS_PATH),
        help=(
            "Corpus to ground queries in. Must match the corpus that is "
            f"indexed and served (default: {DEFAULT_CORPUS_PATH})"
        ),
    )
    parser.add_argument(
        "--max-verbatim-ngram", type=int, default=None,
        help=("Reject and resample queries whose longest verbatim n-gram "
              "overlap with the gold doc >= N tokens. Default: no filtering "
              "(report only). Recommended threshold: 3 or 4."),
    )
    args = parser.parse_args(argv)

    corpus_path = Path(args.corpus)
    print(f"Loading indexed books from {corpus_path}...")
    books = load_indexed_books(corpus_path)
    print(f"Loaded {len(books)} indexed books (with non-empty description)")

    print("Building corpus document-frequency index...")
    corpus_df = build_corpus_df(books)
    print(f"  {len(corpus_df)} unique terms")

    print("Building author index...")
    author_index = build_author_index(books)
    print(f"  {len(author_index)} unique normalized authors")

    mode_label = "DRY-RUN (template)" if args.dry_run else "AZURE OPENAI"
    print(f"\nGenerating {args.n} queries in {mode_label} mode (seed={args.seed})...")
    if args.max_verbatim_ngram is not None:
        print(f"  Leakage filter: rejecting max_verbatim_ngram >= {args.max_verbatim_ngram}")

    queries = generate_query_set(
        books=books,
        n=args.n,
        seed=args.seed,
        dry_run=args.dry_run,
        max_verbatim_ngram=args.max_verbatim_ngram,
        corpus_df=corpus_df,
        author_index=author_index,
    )

    print_stats(queries)

    # Validate gold IDs
    print("\nValidating gold_work_ids against indexed corpus...")
    if validate_gold_ids(queries, books):
        print("  ALL gold_work_ids verified OK")
    else:
        print("  SOME gold_work_ids MISSING -- check output!")

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(queries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved to {out_path}")

    # Show a few samples
    print("\n-- Sample queries --")
    for q in queries[:5]:
        print(f"  [{q['category']}] ({q['difficulty']}) {q['query']!r}")
        gold_count = len(q["gold_work_ids"])
        complete_str = "complete" if q["gold_is_complete"] else "incomplete"
        print(f"    gold: {gold_count} doc(s) [{complete_str}]  "
              f"seed: {q['seed_book']['title']!r}  "
              f"leakage: ngram={q['leakage']['max_verbatim_ngram']}, "
              f"jaccard={q['leakage']['unigram_jaccard']:.3f}")


if __name__ == "__main__":
    main()
