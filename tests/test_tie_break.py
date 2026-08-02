"""Regression tests for deterministic ranking under score ties.

Qdrant's server-side RRF hands back bit-identical scores whenever two documents
sit at the same rank in exactly one input list each, and it breaks those ties by
segment-merge order rather than anything stable. Measured against the deployed
index, two back-to-back identical hybrid requests returned a different rank-1
document on 8 of 40 queries. These tests pin the client-side fix.
"""

import random

from src.qdrant.client import TIE_BREAK_MARGIN, QdrantSearch


def _doc(work_id: str, score: float) -> dict:
    return {"work_id": work_id, "id": work_id, "score": score}


def test_ties_broken_by_work_id():
    """Equal scores must fall back to a stable key, not input order."""
    docs = [_doc("gr:590401645", 0.125), _doc("gr:142310689X", 0.125)]
    assert [d["work_id"] for d in QdrantSearch._stable_rank(docs)] == [
        "gr:142310689X",
        "gr:590401645",
    ]


def test_score_order_dominates_tie_break():
    """The tie-break must only apply within equal scores, never across them."""
    docs = [_doc("aaa", 0.1), _doc("zzz", 0.9), _doc("mmm", 0.5)]
    assert [d["work_id"] for d in QdrantSearch._stable_rank(docs)] == ["zzz", "mmm", "aaa"]


def test_permutations_of_input_produce_identical_output():
    """This is the property that actually failed in production: same set, any
    arrival order, must yield one canonical ranking."""
    # A realistic RRF shape: a few unique scores plus a large tie cluster.
    docs = [_doc(f"gr:{i}", 0.125) for i in range(12)]
    docs += [_doc("gr:top", 1.0), _doc("gr:second", 0.5833334)]

    expected = [d["work_id"] for d in QdrantSearch._stable_rank(docs)]
    rng = random.Random(0)
    for _ in range(50):
        shuffled = docs[:]
        rng.shuffle(shuffled)
        assert [d["work_id"] for d in QdrantSearch._stable_rank(shuffled)] == expected


def test_tie_break_survives_truncation_boundary():
    """A tie straddling the cut must drop the same document every time --
    the failure that changed the returned *set*, not just its order."""
    docs = [_doc("gr:high", 1.0)] + [_doc(f"gr:{i:03d}", 0.125) for i in range(10)]
    rng = random.Random(1)
    expected = [d["work_id"] for d in QdrantSearch._stable_rank(docs)][:5]
    for _ in range(50):
        shuffled = docs[:]
        rng.shuffle(shuffled)
        assert [d["work_id"] for d in QdrantSearch._stable_rank(shuffled)][:5] == expected


def test_missing_work_id_falls_back_to_id():
    """work_id is nullable in the payload; the sort key must not crash on None."""
    docs = [
        {"work_id": None, "id": "bbb", "score": 0.5},
        {"work_id": None, "id": "aaa", "score": 0.5},
    ]
    assert [d["id"] for d in QdrantSearch._stable_rank(docs)] == ["aaa", "bbb"]


def test_margin_leaves_headroom_beyond_requested_window():
    """The fix only works if we actually fetch past the window we return."""
    assert TIE_BREAK_MARGIN > 0
