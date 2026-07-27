"""Tests for src.eval.label -- human labeling CLI for judge validation.

Every test is a proper ``test_*`` function; no assertions at module scope.
Importing this file has no side effects and makes no API calls.
"""

import json
from collections import Counter

import pytest

from src.eval.judge import compute_agreement
from src.eval.label import (
    _already_labeled,
    _grade_of,
    _wrap,
    load_progress,
    run_labeling,
    select_pairs,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def judgments():
    """LLM judgments in eval_redesign.py format (uses `relevance`)."""
    out = {}
    for q in range(4):
        query = f"query {q}"
        entries = []
        for d in range(10):
            # skew like the real corpus: mostly 0s, few 2s
            grade = 0 if d < 7 else (1 if d < 9 else 2)
            entries.append(
                {"work_id": f"W{q}_{d}", "title": f"Book {q}-{d}", "relevance": grade}
            )
        out[query] = entries
    return out


@pytest.fixture
def pooled():
    """Pooled documents keyed by query."""
    return {
        f"query {q}": [
            {
                "work_id": f"W{q}_{d}",
                "title": f"Book {q}-{d}",
                "authors": f"Author {d}",
                "subjects": ["Fiction", "Drama"],
                "description": f"A description of book {d}.",
            }
            for d in range(10)
        ]
        for q in range(4)
    }


def _scripted_keys(monkeypatch, keys):
    """Feed a fixed keypress sequence to the labeling loop."""
    it = iter(keys)
    monkeypatch.setattr("src.eval.label._read_key", lambda: next(it))


# ---------------------------------------------------------------------------
# _grade_of
# ---------------------------------------------------------------------------


def test_grade_of_reads_grade_field():
    assert _grade_of({"grade": 2}) == 2


def test_grade_of_falls_back_to_relevance_field():
    assert _grade_of({"relevance": 1}) == 1


def test_grade_of_prefers_grade_when_both_present():
    assert _grade_of({"grade": 2, "relevance": 0}) == 2


def test_grade_of_returns_none_when_absent():
    assert _grade_of({"work_id": "W1"}) is None


def test_grade_of_preserves_zero():
    """Zero is a valid grade and must not be treated as missing."""
    assert _grade_of({"grade": 0}) == 0


# ---------------------------------------------------------------------------
# select_pairs
# ---------------------------------------------------------------------------


def test_select_pairs_returns_requested_count(judgments, pooled):
    assert len(select_pairs(judgments, pooled, 12, 42, "stratified")) == 12


def test_select_pairs_enriches_with_pooled_metadata(judgments, pooled):
    pair = select_pairs(judgments, pooled, 5, 42, "stratified")[0]
    assert pair["authors"].startswith("Author")
    assert pair["description"]
    assert pair["subjects"] == ["Fiction", "Drama"]


def test_stratified_balances_grades_better_than_random(judgments, pooled):
    strat = Counter(
        p["llm_grade"] for p in select_pairs(judgments, pooled, 12, 42, "stratified")
    )
    rand = Counter(
        p["llm_grade"] for p in select_pairs(judgments, pooled, 12, 42, "random")
    )
    # stratified should surface more of the rare grade-2 pairs
    assert strat[2] >= rand[2]
    assert strat[2] > 0


def test_random_strategy_tracks_natural_prevalence(judgments, pooled):
    pairs = select_pairs(judgments, pooled, 40, 1, "random")
    dist = Counter(p["llm_grade"] for p in pairs)
    # corpus is 70% grade-0, so a representative sample is 0-dominated
    assert dist[0] > dist[1] + dist[2]


def test_select_pairs_is_deterministic_for_a_seed(judgments, pooled):
    a = select_pairs(judgments, pooled, 10, 7, "stratified")
    b = select_pairs(judgments, pooled, 10, 7, "stratified")
    assert [(p["query"], p["work_id"]) for p in a] == [
        (p["query"], p["work_id"]) for p in b
    ]


def test_different_seeds_give_different_samples(judgments, pooled):
    a = select_pairs(judgments, pooled, 10, 1, "stratified")
    b = select_pairs(judgments, pooled, 10, 2, "stratified")
    assert [p["work_id"] for p in a] != [p["work_id"] for p in b]


def test_select_pairs_yields_no_duplicates(judgments, pooled):
    pairs = select_pairs(judgments, pooled, 40, 42, "stratified")
    keys = [(p["query"], p["work_id"]) for p in pairs]
    assert len(keys) == len(set(keys))


def test_select_pairs_skips_docs_missing_from_pool(judgments, pooled):
    judgments["query 0"].append(
        {"work_id": "GHOST", "title": "Not pooled", "relevance": 2}
    )
    pairs = select_pairs(judgments, pooled, 50, 42, "stratified")
    assert all(p["work_id"] != "GHOST" for p in pairs)


def test_select_pairs_handles_n_larger_than_corpus(judgments, pooled):
    pairs = select_pairs(judgments, pooled, 10_000, 42, "stratified")
    assert len(pairs) == 40  # 4 queries x 10 docs


def test_select_pairs_empty_when_no_overlap(judgments):
    assert select_pairs(judgments, {}, 10, 42, "stratified") == []


# ---------------------------------------------------------------------------
# Labeling session
# ---------------------------------------------------------------------------


def test_run_labeling_records_keypresses(tmp_path, judgments, pooled, monkeypatch):
    pairs = select_pairs(judgments, pooled, 3, 42, "stratified")
    _scripted_keys(monkeypatch, ["0", "1", "2"])
    out = tmp_path / "human.json"

    run_labeling(pairs, out, "stratified")

    data = json.loads(out.read_text(encoding="utf-8"))
    assert sorted(e["grade"] for v in data.values() for e in v) == [0, 1, 2]


def test_run_labeling_writes_both_grade_and_relevance(
    tmp_path, judgments, pooled, monkeypatch
):
    """`grade` is read by judge.py, `relevance` by metrics.py."""
    pairs = select_pairs(judgments, pooled, 1, 42, "stratified")
    _scripted_keys(monkeypatch, ["2"])
    out = tmp_path / "human.json"

    run_labeling(pairs, out, "stratified")

    entry = next(iter(json.loads(out.read_text(encoding="utf-8")).values()))[0]
    assert entry["grade"] == entry["relevance"] == 2


def test_quit_stops_early_but_keeps_progress(
    tmp_path, judgments, pooled, monkeypatch
):
    pairs = select_pairs(judgments, pooled, 5, 42, "stratified")
    _scripted_keys(monkeypatch, ["1", "2", "q"])
    out = tmp_path / "human.json"

    run_labeling(pairs, out, "stratified")

    data = json.loads(out.read_text(encoding="utf-8"))
    assert sum(len(v) for v in data.values()) == 2


def test_skip_does_not_record_a_label(tmp_path, judgments, pooled, monkeypatch):
    pairs = select_pairs(judgments, pooled, 3, 42, "stratified")
    _scripted_keys(monkeypatch, ["s", "s", "1"])
    out = tmp_path / "human.json"

    run_labeling(pairs, out, "stratified")

    data = json.loads(out.read_text(encoding="utf-8"))
    assert sum(len(v) for v in data.values()) == 1


def test_back_removes_the_previous_label(tmp_path, judgments, pooled, monkeypatch):
    pairs = select_pairs(judgments, pooled, 3, 42, "stratified")
    # label, go back, relabel differently, then finish
    _scripted_keys(monkeypatch, ["0", "b", "2", "1", "1"])
    out = tmp_path / "human.json"

    run_labeling(pairs, out, "stratified")

    data = json.loads(out.read_text(encoding="utf-8"))
    first = pairs[0]
    entry = [e for e in data[first["query"]] if e["work_id"] == first["work_id"]][0]
    assert entry["grade"] == 2  # corrected value, not the original 0


def test_back_on_first_item_is_a_noop(tmp_path, judgments, pooled, monkeypatch):
    pairs = select_pairs(judgments, pooled, 2, 42, "stratified")
    _scripted_keys(monkeypatch, ["b", "1", "2"])
    out = tmp_path / "human.json"

    run_labeling(pairs, out, "stratified")

    data = json.loads(out.read_text(encoding="utf-8"))
    assert sum(len(v) for v in data.values()) == 2


def test_session_is_resumable(tmp_path, judgments, pooled, monkeypatch):
    pairs = select_pairs(judgments, pooled, 4, 42, "stratified")
    out = tmp_path / "human.json"

    _scripted_keys(monkeypatch, ["1", "q"])
    run_labeling(pairs, out, "stratified")
    assert sum(len(v) for v in load_progress(out).values()) == 1

    _scripted_keys(monkeypatch, ["2", "0", "1"])
    run_labeling(pairs, out, "stratified")
    assert sum(len(v) for v in load_progress(out).values()) == 4


def test_resume_does_not_duplicate_labels(tmp_path, judgments, pooled, monkeypatch):
    pairs = select_pairs(judgments, pooled, 3, 42, "stratified")
    out = tmp_path / "human.json"

    _scripted_keys(monkeypatch, ["1", "1", "1"])
    run_labeling(pairs, out, "stratified")
    # second run has nothing left to do and must not prompt
    run_labeling(pairs, out, "stratified")

    data = load_progress(out)
    keys = [(q, e["work_id"]) for q, v in data.items() for e in v]
    assert len(keys) == len(set(keys)) == 3


def test_writes_sampling_metadata_sidecar(
    tmp_path, judgments, pooled, monkeypatch
):
    pairs = select_pairs(judgments, pooled, 2, 42, "stratified")
    _scripted_keys(monkeypatch, ["1", "2"])
    out = tmp_path / "human.json"

    run_labeling(pairs, out, "stratified")

    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["sampling_strategy"] == "stratified"
    assert meta["n_labeled"] == 2
    assert "kappa" in meta["note"].lower()


def test_already_labeled_builds_pair_keys():
    human = {"q1": [{"work_id": "W1", "grade": 1}, {"work_id": "W2", "grade": 0}]}
    assert _already_labeled(human) == {("q1", "W1"), ("q1", "W2")}


def test_load_progress_returns_empty_for_missing_file(tmp_path):
    assert load_progress(tmp_path / "nope.json") == {}


# ---------------------------------------------------------------------------
# Integration with the agreement scorer
# ---------------------------------------------------------------------------


def test_output_is_scorable_against_llm_judgments(
    tmp_path, judgments, pooled, monkeypatch
):
    """The whole point: human.json must feed compute_agreement directly."""
    pairs = select_pairs(judgments, pooled, 6, 42, "stratified")
    _scripted_keys(monkeypatch, [str(p["llm_grade"]) for p in pairs])
    out = tmp_path / "human.json"

    run_labeling(pairs, out, "stratified")

    stats = compute_agreement(judgments, load_progress(out))
    assert stats["n"] == 6
    assert stats["kappa"] == 1.0  # labeler mirrored the LLM exactly


def test_total_disagreement_scores_negative_kappa(
    tmp_path, judgments, pooled, monkeypatch
):
    pairs = select_pairs(judgments, pooled, 6, 42, "stratified")
    # always pick a different grade than the LLM
    _scripted_keys(
        monkeypatch, [str((p["llm_grade"] + 1) % 3) for p in pairs]
    )
    out = tmp_path / "human.json"

    run_labeling(pairs, out, "stratified")

    stats = compute_agreement(judgments, load_progress(out))
    assert stats["raw_agreement"] == 0.0
    assert stats["kappa"] < 0


# ---------------------------------------------------------------------------
# Text wrapping
# ---------------------------------------------------------------------------


def test_wrap_respects_width():
    lines = _wrap("word " * 60, width=30, indent="  ", max_lines=10)
    assert all(len(line) <= 34 for line in lines)


def test_wrap_truncates_at_max_lines():
    lines = _wrap("word " * 500, width=20, indent="", max_lines=3)
    assert len(lines) == 3
    assert lines[-1].endswith("...")


def test_wrap_handles_empty_string():
    assert _wrap("", width=20, indent="", max_lines=3) == []
