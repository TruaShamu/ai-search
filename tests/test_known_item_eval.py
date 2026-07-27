"""Tests for the known-item evaluation module.

These are unit tests that validate dataset generation, metric computation,
baseline management, and title-word classification without calling the live API.
"""

import json
import random
from pathlib import Path
from unittest.mock import patch

import pytest

from src.eval.known_items import (
    _find_unique_titles,
    _is_distinctive,
    _make_partial,
    _make_typo,
    _tokenize_title,
    classify_title,
    compute_corpus_word_df,
    generate_known_items,
    load_known_items,
    save_known_items,
)
from src.eval.known_item_eval import (
    _eval_items,
    _extract_ids,
    _print_keyword_split,
    _print_standard_report,
    load_baseline,
    save_baseline,
)


# ---------------------------------------------------------------------------
# known_items.py tests
# ---------------------------------------------------------------------------


class TestDistinctiveness:
    def test_short_titles_rejected(self):
        assert not _is_distinctive("Go")
        assert not _is_distinctive("The Cat")

    def test_long_titles_accepted(self):
        assert _is_distinctive("The Adventures of Tom Sawyer")
        assert _is_distinctive("A Brief History of Time")

    def test_allcaps_rejected(self):
        assert not _is_distinctive("INTRODUCTION TO PSYCHOLOGY")

    def test_mixed_case_accepted(self):
        assert _is_distinctive("Introduction to Psychology")


class TestUniqueFilter:
    def test_removes_duplicates(self):
        books = [
            {"title": "Hamlet", "work_id": "1"},
            {"title": "hamlet", "work_id": "2"},  # case-insensitive dup
            {"title": "Pride and Prejudice", "work_id": "3"},
        ]
        unique = _find_unique_titles(books)
        assert len(unique) == 1
        assert unique[0]["title"] == "Pride and Prejudice"


class TestTypoGeneration:
    def test_produces_different_string(self):
        rng = random.Random(42)
        title = "The Great Gatsby"
        result = _make_typo(title, rng)
        assert result != title
        assert len(result) == len(title)

    def test_single_word_unchanged(self):
        rng = random.Random(42)
        assert _make_typo("Go", rng) == "Go"


class TestPartialGeneration:
    def test_colon_split(self):
        rng = random.Random(42)
        result = _make_partial("Harry Potter: The Sorcerer's Stone", rng)
        assert result == "Harry Potter"

    def test_word_truncation(self):
        rng = random.Random(42)
        result = _make_partial("One Two Three Four Five", rng)
        assert "One" in result
        assert len(result.split()) < 5


class TestTokenize:
    def test_removes_stopwords(self):
        tokens = _tokenize_title("The Art of War")
        assert "the" not in tokens
        assert "of" not in tokens
        assert "art" in tokens
        assert "war" in tokens

    def test_removes_punctuation(self):
        tokens = _tokenize_title("Hello, World!")
        assert "hello" in tokens
        assert "world" in tokens

    def test_removes_single_chars(self):
        tokens = _tokenize_title("A B C dragon")
        assert "dragon" in tokens
        assert "b" not in tokens


class TestTitleWordClassification:
    """Test corpus-DF-based title classification."""

    @pytest.fixture()
    def corpus(self):
        """Synthetic corpus for DF testing.  1000+ docs so thresholds work."""
        books = []
        for i in range(1000):
            books.append({
                "title": f"Book {i}",
                "description": "A love story about people and life.",
            })
        # Add a few with a rare word
        for i in range(3):
            books.append({
                "title": f"Xyzzy volume {i}",
                "description": "The xyzzy chronicles. Rare word here.",
            })
        return books

    def test_common_words_classified(self, corpus):
        df, total = compute_corpus_word_df(corpus)
        # "love" and "story" appear in 1000/1003 docs
        assert classify_title("A love story", df, total) == "common_words"

    def test_distinctive_words_classified(self, corpus):
        df, total = compute_corpus_word_df(corpus)
        # "xyzzy" appears in 3/1003 docs (0.3%) -- distinctive at 0.5%
        assert classify_title("The xyzzy chronicles", df, total) == "distinctive"

    def test_mixed_title_is_distinctive(self, corpus):
        df, total = compute_corpus_word_df(corpus)
        assert classify_title("A love xyzzy story", df, total) == "distinctive"

    def test_empty_title(self, corpus):
        df, total = compute_corpus_word_df(corpus)
        assert classify_title("", df, total) == "common_words"


class TestDeterminism:
    """Verify that generation with same seed produces identical output."""

    def test_same_seed_same_output(self):
        from src.eval.known_items import _DEFAULT_CORPUS
        if not _DEFAULT_CORPUS.exists():
            pytest.skip("Corpus not available")

        items1, hv1 = generate_known_items(count=10, seed=99)
        items2, hv2 = generate_known_items(count=10, seed=99)

        assert items1 == items2
        assert hv1 == hv2

    def test_different_seed_different_output(self):
        from src.eval.known_items import _DEFAULT_CORPUS
        if not _DEFAULT_CORPUS.exists():
            pytest.skip("Corpus not available")

        items1, _ = generate_known_items(count=10, seed=1)
        items2, _ = generate_known_items(count=10, seed=2)

        titles1 = {i["title"] for i in items1}
        titles2 = {i["title"] for i in items2}
        assert titles1 != titles2

    def test_items_have_title_word_class(self):
        from src.eval.known_items import _DEFAULT_CORPUS
        if not _DEFAULT_CORPUS.exists():
            pytest.skip("Corpus not available")

        items, _ = generate_known_items(count=10, seed=42)
        for item in items:
            assert "title_word_class" in item
            assert item["title_word_class"] in ("distinctive", "common_words")


class TestSaveAndLoad:
    def test_roundtrip(self, tmp_path):
        items = [{"query": "Test Book Title Here", "work_id": "OL1W",
                  "title": "Test Book Title Here", "authors": ["Author"],
                  "title_word_class": "distinctive"}]
        hard = [{"query": "Test Book", "work_id": "OL1W",
                 "title": "Test Book Title Here", "variant_type": "partial_title"}]

        ip, vp = save_known_items(items, hard, output_dir=tmp_path)
        loaded_items = load_known_items(ip)
        assert loaded_items == items

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_known_items(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# known_item_eval.py tests
# ---------------------------------------------------------------------------


class TestExtractIds:
    def test_strips_prefix(self):
        results = [
            {"work_id": "/works/OL123W"},
            {"work_id": "/works/OL456W"},
        ]
        assert _extract_ids(results) == ["OL123W", "OL456W"]

    def test_bare_ids_unchanged(self):
        results = [{"work_id": "OL789W"}]
        assert _extract_ids(results) == ["OL789W"]


class TestEvalMetrics:
    """Test metric computation with mocked API responses."""

    def _mock_search(self, api_url, query, mode, top_k=10):
        return {
            "results": [
                {"work_id": "/works/OL1W", "title": "Book One", "score": 1.0},
                {"work_id": "/works/OL2W", "title": "Book Two", "score": 0.5},
            ]
        }

    def _mock_search_rank2(self, api_url, query, mode, top_k=10):
        return {
            "results": [
                {"work_id": "/works/OL999W", "title": "Other", "score": 1.0},
                {"work_id": "/works/OL1W", "title": "Book One", "score": 0.5},
            ]
        }

    @patch("src.eval.known_item_eval._search")
    def test_perfect_accuracy(self, mock_search):
        mock_search.side_effect = self._mock_search
        items = [{"query": "Book One", "work_id": "OL1W", "title": "Book One"}]
        results = _eval_items("http://fake", items, modes=("hybrid",))
        assert results["hybrid"]["accuracy_at_1"] == 1.0
        assert results["hybrid"]["mrr"] == 1.0

    @patch("src.eval.known_item_eval._search")
    def test_rank2_metrics(self, mock_search):
        mock_search.side_effect = self._mock_search_rank2
        items = [{"query": "Book One", "work_id": "OL1W", "title": "Book One"}]
        results = _eval_items("http://fake", items, modes=("hybrid",))
        assert results["hybrid"]["accuracy_at_1"] == 0.0
        assert results["hybrid"]["accuracy_at_5"] == 1.0
        assert results["hybrid"]["mrr"] == 0.5

    @patch("src.eval.known_item_eval._search")
    def test_api_error_counted_as_miss(self, mock_search):
        mock_search.side_effect = Exception("timeout")
        items = [{"query": "Book One", "work_id": "OL1W", "title": "Book One"}]
        results = _eval_items("http://fake", items, modes=("keyword",))
        assert results["keyword"]["accuracy_at_1"] == 0.0
        assert results["keyword"]["mrr"] == 0.0
        assert results["keyword"]["details"][0].get("error") == "timeout"

    @patch("src.eval.known_item_eval._search")
    def test_title_word_class_preserved_in_details(self, mock_search):
        mock_search.side_effect = self._mock_search
        items = [{"query": "Book One", "work_id": "OL1W", "title": "Book One",
                  "title_word_class": "distinctive"}]
        results = _eval_items("http://fake", items, modes=("keyword",))
        detail = results["keyword"]["details"][0]
        assert detail["title_word_class"] == "distinctive"


class TestBaseline:
    """Test baseline save/load cycle and gate logic."""

    def test_save_and_load(self, tmp_path):
        results = {
            "hybrid": {"accuracy_at_1": 0.94, "accuracy_at_5": 1.0, "mrr": 0.97,
                       "hits_at_1": 47, "hits_at_5": 50, "total": 50, "details": []},
        }
        bl_path = save_baseline(
            results, api_url="http://test", item_count=50,
            path=tmp_path / "baseline.json",
        )
        loaded = load_baseline(bl_path)
        assert loaded is not None
        assert loaded["modes"]["hybrid"]["accuracy_at_1"] == 0.94
        assert loaded["item_count"] == 50
        assert loaded["gated_modes"] == ["hybrid"]
        assert "created_at" in loaded

    def test_load_missing_returns_none(self, tmp_path):
        assert load_baseline(tmp_path / "nope.json") is None

    def test_gate_passes_within_threshold(self):
        """Verify gate passes when drop is within max_drop_pp."""
        results = {
            "hybrid": {"accuracy_at_1": 0.90, "accuracy_at_5": 1.0, "mrr": 0.95,
                       "hits_at_1": 45, "hits_at_5": 50, "total": 50, "details": []},
        }
        baseline = {
            "created_at": "2026-01-01",
            "git_sha": "abc",
            "modes": {"hybrid": {"accuracy_at_1": 0.94}},
        }
        # Drop = -4pp, max = 5pp, should PASS
        passed = _print_standard_report(
            results, baseline, max_drop_pp=5, gated_modes=["hybrid"]
        )
        assert passed is True

    def test_gate_fails_beyond_threshold(self):
        """Verify gate fails when drop exceeds max_drop_pp."""
        results = {
            "hybrid": {"accuracy_at_1": 0.80, "accuracy_at_5": 0.9, "mrr": 0.85,
                       "hits_at_1": 40, "hits_at_5": 45, "total": 50, "details": []},
        }
        baseline = {
            "created_at": "2026-01-01",
            "git_sha": "abc",
            "modes": {"hybrid": {"accuracy_at_1": 0.94}},
        }
        # Drop = -14pp, max = 5pp, should FAIL
        passed = _print_standard_report(
            results, baseline, max_drop_pp=5, gated_modes=["hybrid"]
        )
        assert passed is False

    def test_non_gated_modes_dont_affect_gate(self):
        """Keyword and vector are diagnostic and never fail the gate."""
        results = {
            "keyword": {"accuracy_at_1": 0.50, "accuracy_at_5": 0.7, "mrr": 0.6,
                        "hits_at_1": 25, "hits_at_5": 35, "total": 50, "details": []},
            "hybrid": {"accuracy_at_1": 0.94, "accuracy_at_5": 1.0, "mrr": 0.97,
                       "hits_at_1": 47, "hits_at_5": 50, "total": 50, "details": []},
        }
        baseline = {
            "created_at": "2026-01-01",
            "git_sha": "abc",
            "modes": {
                "keyword": {"accuracy_at_1": 0.90},
                "hybrid": {"accuracy_at_1": 0.94},
            },
        }
        # keyword drops -40pp but isn't gated -- should still PASS
        passed = _print_standard_report(
            results, baseline, max_drop_pp=5, gated_modes=["hybrid"]
        )
        assert passed is True
