"""Tests for corpus-grounded query generation (src.eval.query_gen)."""

import random

import pytest

from src.eval.query_gen import (
    CATEGORY_WEIGHTS,
    _add_typo,
    _broad_genre,
    _book_searchable_text,
    _desc_len_bucket,
    _max_verbatim_ngram_len,
    _normalize_author,
    _normalize_for_dedup,
    _partial_title,
    _tokenize,
    _tokenize_no_stop,
    assign_categories,
    build_author_index,
    build_corpus_df,
    estimate_difficulty,
    generate_dry_run_query,
    generate_query_set,
    gold_is_complete,
    is_near_duplicate,
    lexical_overlap,
    sample_seed_books,
    validate_gold_ids,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_books():
    """A small set of synthetic books for unit tests."""
    return [
        {
            "work_id": f"/works/OL{i}W",
            "title": title,
            "authors": authors,
            "description": desc,
            "subjects": subjects,
            "first_publish_year": year,
            "tier": 1,
        }
        for i, (title, authors, desc, subjects, year) in enumerate([
            ("The Great Adventure", ["Jane Doe"], "A thrilling tale of exploration and courage in uncharted wilderness.", ["Fiction", "Adventure"], 2005),
            ("Quantum Physics Explained", ["Dr. Smith"], "An introduction to quantum mechanics for beginners and students.", ["Science", "Physics", "Education"], 1998),
            ("Cooking with Herbs", ["Chef Mario"], "A comprehensive guide to using fresh herbs in every meal.", ["Cooking", "Food"], 2010),
            ("Mystery at Midnight", ["A.B. Clark"], "A detective investigates a series of baffling crimes in London.", ["Fiction", "Mystery"], 1985),
            ("Children's Garden", ["Emily Rose"], "A charming story for young readers about growing things.", ["Children's Fiction", "Juvenile Literature"], 2015),
            ("Poetry of the Soul", ["Robert Verse"], "A collection of modern poems about love and loss.", ["Poetry"], None),
            ("History of Rome", ["Prof. Augustus"], "The rise and fall of the Roman empire across centuries.", ["History", "Ancient Civilizations"], 1920),
            ("Business Strategy", ["MBA Guy"], "How to build a successful business from the ground up.", ["Business", "Economics"], 2001),
            ("Another Adventure Book", ["Jane Doe"], "A second thrilling adventure by Jane Doe in the Arctic.", ["Fiction", "Adventure"], 2008),
        ])
    ]


# ── Tokenization tests ─────────────────────────────────────────────────────

class TestTokenization:
    def test_tokenize_basic(self):
        assert _tokenize("Hello, World! 123") == ["hello", "world", "123"]

    def test_tokenize_no_stop_removes_stopwords(self):
        result = _tokenize_no_stop("a book about the history of Rome")
        assert "a" not in result
        assert "the" not in result
        assert "of" not in result
        assert "history" in result
        assert "rome" in result

    def test_tokenize_no_stop_removes_book_words(self):
        result = _tokenize_no_stop("novel about adventure books")
        assert "novel" not in result
        assert "books" not in result
        assert "adventure" in result


# ── Corpus infrastructure tests ─────────────────────────────────────────────

class TestCorpusInfra:
    def test_book_searchable_text(self, sample_books):
        text = _book_searchable_text(sample_books[0])
        assert "The Great Adventure" in text
        assert "Jane Doe" in text
        assert "Fiction" in text
        assert "thrilling" in text

    def test_build_corpus_df(self, sample_books):
        df = build_corpus_df(sample_books)
        assert isinstance(df, dict)
        assert df.get("fiction", 0) >= 2  # appears in multiple books
        assert df.get("quantum", 0) >= 1

    def test_normalize_author(self):
        assert _normalize_author("Dr. Jane Smith") == "jane smith"
        assert _normalize_author("Prof. Augustus") == "augustus"
        assert _normalize_author("  John   Doe  ") == "john doe"
        assert _normalize_author("Sir Arthur Conan Doyle") == "arthur conan doyle"

    def test_build_author_index(self, sample_books):
        idx = build_author_index(sample_books)
        assert "jane doe" in idx
        # Jane Doe has two books (indices 0 and 8)
        assert len(idx["jane doe"]) == 2
        assert "/works/OL0W" in idx["jane doe"]
        assert "/works/OL8W" in idx["jane doe"]

    def test_build_author_index_strips_title(self, sample_books):
        idx = build_author_index(sample_books)
        # "Dr. Smith" normalizes to "smith"
        assert "smith" in idx
        # "Prof. Augustus" normalizes to "augustus"
        assert "augustus" in idx


# ── Lexical overlap tests ───────────────────────────────────────────────────

class TestLexicalOverlap:
    def test_max_verbatim_ngram_exact_match(self):
        q = ["the", "great", "adventure"]
        d = ["a", "the", "great", "adventure", "story"]
        assert _max_verbatim_ngram_len(q, d) == 3

    def test_max_verbatim_ngram_no_match(self):
        q = ["quantum", "physics"]
        d = ["cooking", "with", "herbs"]
        assert _max_verbatim_ngram_len(q, d) == 0

    def test_max_verbatim_ngram_partial(self):
        q = ["great", "adventure", "novel"]
        d = ["the", "great", "adventure", "of", "life"]
        assert _max_verbatim_ngram_len(q, d) == 2

    def test_max_verbatim_ngram_empty(self):
        assert _max_verbatim_ngram_len([], ["a", "b"]) == 0
        assert _max_verbatim_ngram_len(["a"], []) == 0

    def test_lexical_overlap_returns_all_keys(self, sample_books):
        df = build_corpus_df(sample_books)
        result = lexical_overlap("thrilling adventure book", sample_books[0], df, len(sample_books))
        assert "unigram_jaccard" in result
        assert "max_verbatim_ngram" in result
        assert "rare_term_overlap" in result

    def test_lexical_overlap_high_for_title_copy(self, sample_books):
        df = build_corpus_df(sample_books)
        result = lexical_overlap("The Great Adventure", sample_books[0], df, len(sample_books))
        assert result["unigram_jaccard"] > 0
        assert result["max_verbatim_ngram"] >= 2

    def test_lexical_overlap_low_for_unrelated(self, sample_books):
        df = build_corpus_df(sample_books)
        result = lexical_overlap("quantum physics", sample_books[2], df, len(sample_books))
        assert result["unigram_jaccard"] < 0.1


# ── Helper tests ────────────────────────────────────────────────────────────

class TestHelpers:
    def test_broad_genre_fiction(self, sample_books):
        assert _broad_genre(sample_books[0]) == "fiction"

    def test_broad_genre_science(self, sample_books):
        assert _broad_genre(sample_books[1]) == "science"

    def test_broad_genre_cooking(self, sample_books):
        assert _broad_genre(sample_books[2]) == "cooking"

    def test_desc_len_bucket_short(self):
        assert _desc_len_bucket({"description": "x" * 50}) == "short"

    def test_desc_len_bucket_medium(self):
        assert _desc_len_bucket({"description": "x" * 500}) == "medium"

    def test_desc_len_bucket_long(self):
        assert _desc_len_bucket({"description": "x" * 1000}) == "long"

    def test_add_typo_changes_string(self):
        rng = random.Random(42)
        title = "The Great Gatsby"
        result = _add_typo(title, rng)
        assert result != title
        assert len(result) == len(title)

    def test_partial_title_drops_subtitle(self):
        rng = random.Random(42)
        assert _partial_title("Main Title: A Subtitle", rng) == "Main Title"

    def test_partial_title_shortens(self):
        rng = random.Random(42)
        result = _partial_title("One Two Three Four Five", rng)
        assert len(result.split()) < 5

    def test_normalize_for_dedup(self):
        assert _normalize_for_dedup("Hello, World!") == "hello world"
        assert _normalize_for_dedup("  spaces   here  ") == "spaces here"


class TestNearDuplicate:
    def test_identical_is_duplicate(self):
        assert is_near_duplicate("books by author", ["books by author"])

    def test_similar_is_duplicate(self):
        assert is_near_duplicate("books by jane", ["books by jane doe"], threshold=0.7)

    def test_different_is_not_duplicate(self):
        assert not is_near_duplicate("quantum physics", ["cooking recipes"])


class TestCategoryAssignment:
    def test_correct_count(self):
        cats = assign_categories(100, random.Random(42))
        assert len(cats) == 100

    def test_all_valid_categories(self):
        cats = assign_categories(50, random.Random(42))
        for c in cats:
            assert c in CATEGORY_WEIGHTS

    def test_distribution_roughly_correct(self):
        cats = assign_categories(200, random.Random(42))
        from collections import Counter
        counts = Counter(cats)
        for cat, weight in CATEGORY_WEIGHTS.items():
            expected = 200 * weight
            assert abs(counts[cat] - expected) <= 5, f"{cat}: {counts[cat]} vs {expected}"


class TestSampling:
    def test_sample_returns_correct_count(self, sample_books):
        seeds = sample_seed_books(sample_books, 5, random.Random(42))
        assert len(seeds) == 5

    def test_sample_no_duplicates(self, sample_books):
        seeds = sample_seed_books(sample_books, 8, random.Random(42))
        ids = [b["work_id"] for b in seeds]
        assert len(ids) == len(set(ids))

    def test_sample_reproducible(self, sample_books):
        s1 = sample_seed_books(sample_books, 5, random.Random(42))
        s2 = sample_seed_books(sample_books, 5, random.Random(42))
        assert [b["work_id"] for b in s1] == [b["work_id"] for b in s2]


class TestDryRunGeneration:
    def test_title_lookup(self, sample_books):
        rng = random.Random(42)
        q = generate_dry_run_query(sample_books[0], "title_lookup", rng)
        assert isinstance(q, str) and len(q) > 0

    def test_author(self, sample_books):
        rng = random.Random(42)
        q = generate_dry_run_query(sample_books[0], "author", rng)
        assert "jane" in q.lower() or "doe" in q.lower()

    def test_genre_topic(self, sample_books):
        rng = random.Random(42)
        q = generate_dry_run_query(sample_books[0], "genre_topic", rng)
        assert len(q) > 3

    def test_exploratory(self, sample_books):
        rng = random.Random(42)
        q = generate_dry_run_query(sample_books[0], "exploratory", rng)
        assert len(q) > 5

    def test_combined(self, sample_books):
        rng = random.Random(42)
        q = generate_dry_run_query(sample_books[0], "combined", rng)
        assert len(q) > 3


class TestDifficulty:
    def test_exact_title_is_easy(self, sample_books):
        d = estimate_difficulty("The Great Adventure", sample_books[0], "title_lookup")
        assert d == "easy"

    def test_partial_title_is_medium(self, sample_books):
        d = estimate_difficulty("Great Adventure", sample_books[0], "title_lookup")
        assert d == "medium"

    def test_exploratory_is_hard(self, sample_books):
        d = estimate_difficulty("something thematic", sample_books[0], "exploratory")
        assert d == "hard"

    def test_author_full_match_is_easy(self, sample_books):
        d = estimate_difficulty("books by Jane Doe", sample_books[0], "author")
        assert d == "easy"

    def test_author_last_name_is_medium(self, sample_books):
        d = estimate_difficulty("books by Doe", sample_books[0], "author")
        assert d == "medium"

    def test_combined_with_subject_is_medium(self, sample_books):
        d = estimate_difficulty("fiction published around 2005", sample_books[0], "combined")
        assert d in ("easy", "medium")


class TestGoldCompleteness:
    def test_title_lookup_complete(self):
        assert gold_is_complete("title_lookup") is True

    def test_author_complete(self):
        assert gold_is_complete("author") is True

    def test_genre_topic_incomplete(self):
        assert gold_is_complete("genre_topic") is False

    def test_exploratory_incomplete(self):
        assert gold_is_complete("exploratory") is False

    def test_combined_incomplete(self):
        assert gold_is_complete("combined") is False


class TestGenerateQuerySet:
    def test_dry_run_produces_correct_count(self, sample_books):
        queries = generate_query_set(sample_books, n=5, seed=42, dry_run=True)
        assert len(queries) == 5

    def test_dry_run_has_required_keys(self, sample_books):
        queries = generate_query_set(sample_books, n=3, seed=42, dry_run=True)
        for q in queries:
            assert "query" in q
            assert "category" in q
            assert "gold_work_ids" in q
            assert "gold_is_complete" in q
            assert "difficulty" in q
            assert "leakage" in q
            assert "seed_book" in q
            assert isinstance(q["gold_work_ids"], list)
            assert len(q["gold_work_ids"]) >= 1

    def test_leakage_keys_present(self, sample_books):
        queries = generate_query_set(sample_books, n=3, seed=42, dry_run=True)
        for q in queries:
            leak = q["leakage"]
            assert "unigram_jaccard" in leak
            assert "max_verbatim_ngram" in leak
            assert "rare_term_overlap" in leak

    def test_dry_run_reproducible(self, sample_books):
        q1 = generate_query_set(sample_books, n=5, seed=42, dry_run=True)
        q2 = generate_query_set(sample_books, n=5, seed=42, dry_run=True)
        assert [q["query"] for q in q1] == [q["query"] for q in q2]

    def test_author_queries_have_multiple_gold_ids(self, sample_books):
        """Jane Doe has 2 books; author queries for her should include both."""
        queries = generate_query_set(sample_books, n=8, seed=42, dry_run=True)
        author_qs = [q for q in queries if q["category"] == "author"]
        # Check any author query for Jane Doe has 2 gold IDs
        jane_qs = [q for q in author_qs
                    if "jane" in q["query"].lower() or "doe" in q["query"].lower()]
        for q in jane_qs:
            assert len(q["gold_work_ids"]) == 2, (
                f"Expected 2 gold IDs for Jane Doe query, got {len(q['gold_work_ids'])}"
            )

    def test_gold_is_complete_flag(self, sample_books):
        queries = generate_query_set(sample_books, n=8, seed=42, dry_run=True)
        for q in queries:
            expected = q["category"] in ("title_lookup", "author")
            assert q["gold_is_complete"] == expected

    def test_max_verbatim_ngram_filter(self, sample_books):
        """With a strict filter, some queries should be rejected."""
        # N=1 is extremely strict -- almost everything gets filtered
        q_strict = generate_query_set(
            sample_books, n=5, seed=42, dry_run=True, max_verbatim_ngram=1
        )
        q_normal = generate_query_set(
            sample_books, n=5, seed=42, dry_run=True, max_verbatim_ngram=None
        )
        # The strict set should have different queries (resampled)
        # or potentially fewer if it can't fill
        assert len(q_strict) <= len(q_normal)

    def test_validate_gold_ids_pass(self, sample_books):
        queries = generate_query_set(sample_books, n=5, seed=42, dry_run=True)
        assert validate_gold_ids(queries, sample_books)

    def test_validate_gold_ids_fail(self, sample_books):
        queries = [{"query": "fake query", "gold_work_ids": ["/works/OLNONEXISTENTW"]}]
        assert not validate_gold_ids(queries, sample_books)
