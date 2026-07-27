"""Tests for src.eval.judge -- calibrated LLM-as-judge with escalation.

Every test is a proper ``test_*`` function; no assertions at module scope.
Importing this file has no side effects.
"""

import csv
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest

from src.eval.judge import (
    AzureJudgeClient,
    _detect_contradiction,
    _has_strict_majority,
    _parse_judge_response,
    cohens_kappa,
    compute_agreement,
    export_audit_csv,
    grade_distribution,
    judge_pair,
    krippendorffs_alpha,
    self_consistency_report,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def dry_run_client():
    """AzureJudgeClient in dry-run mode (no API calls)."""
    return AzureJudgeClient(dry_run=True)


def _find_three_way_split_query(work_id: str = "OL_SPLIT", limit: int = 1000) -> str:
    """Search for a (query, work_id) that produces a 3-way split under
    dry-run hash-based synthetic grading.  Python's ``hash()`` is session-
    randomized, so we search at runtime."""
    for qi in range(limit):
        candidate = f"split_{qi}"
        samps = [hash(f"{candidate}|{work_id}|{i}") % 3 for i in range(3)]
        if len(set(samps)) == 3:
            return candidate
    pytest.fail(f"Could not find a 3-way split combo in {limit} tries")


# ===================================================================
# Cohen's kappa
# ===================================================================

class TestCohensKappa:
    """Cohen's kappa implementation verified against hand computation.

    Rater A: [0,0,1,1,2,2,0,1,2,0]   Rater B: [0,1,1,1,2,0,0,2,2,0]
    Agreements at indices 0,2,3,4,6,8,9 -> 7/10  => p_o = 0.7
    Marginals: A(0)=4 A(1)=3 A(2)=3   B(0)=4 B(1)=3 B(2)=3
    p_e = (4/10)^2 + (3/10)^2 + (3/10)^2 = 0.34
    kappa = (0.7 - 0.34)/(1 - 0.34) = 0.36/0.66 = 0.545454...
    """

    A = [0, 0, 1, 1, 2, 2, 0, 1, 2, 0]
    B = [0, 1, 1, 1, 2, 0, 0, 2, 2, 0]

    def test_matches_hand_computation(self):
        expected = 0.36 / 0.66
        got = cohens_kappa(self.A, self.B)
        assert abs(got - expected) < 1e-4, f"{got} != {expected}"

    def test_perfect_agreement(self):
        assert cohens_kappa([0, 1, 2, 0], [0, 1, 2, 0]) == 1.0

    def test_anti_agreement_is_negative(self):
        k = cohens_kappa([0, 0, 1, 1], [1, 1, 0, 0])
        assert k < 0, f"Expected negative kappa, got {k}"


# ===================================================================
# Krippendorff's alpha
# ===================================================================

class TestKrippendorffsAlpha:

    A = [0, 0, 1, 1, 2, 2, 0, 1, 2, 0]
    B = [0, 1, 1, 1, 2, 0, 0, 2, 2, 0]

    def test_in_reasonable_range(self):
        alpha = krippendorffs_alpha(self.A, self.B)
        assert 0.3 < alpha < 0.7, f"alpha={alpha} out of range"

    def test_perfect_agreement(self):
        assert krippendorffs_alpha([0, 1, 2], [0, 1, 2]) == 1.0


# ===================================================================
# compute_agreement (structured JSON-like inputs)
# ===================================================================

class TestComputeAgreement:

    def test_overlapping_judgments(self):
        a = {"q1": [
            {"work_id": "w1", "grade": 0},
            {"work_id": "w2", "grade": 1},
            {"work_id": "w3", "grade": 2},
        ]}
        b = {"q1": [
            {"work_id": "w1", "grade": 0},
            {"work_id": "w2", "grade": 1},
            {"work_id": "w3", "grade": 1},
        ]}
        stats = compute_agreement(a, b)
        assert stats["n"] == 3
        assert stats["raw_agreement"] == pytest.approx(2 / 3, abs=1e-4)


# ===================================================================
# Response parsing
# ===================================================================

class TestParseJudgeResponse:

    def test_standard_json(self):
        g, r = _parse_judge_response('{"grade": 2, "reasoning": "perfect match"}')
        assert g == 2
        assert "perfect" in r

    def test_fenced_json(self):
        g, _ = _parse_judge_response('```json\n{"grade": 1, "reasoning": "ok"}\n```')
        assert g == 1

    def test_empty_response_returns_none(self):
        g, _ = _parse_judge_response("")
        assert g is None, "empty response must yield None, not 0"

    def test_invalid_grade_returns_none(self):
        g, _ = _parse_judge_response('{"grade": 5, "reasoning": "oops"}')
        assert g is None


# ===================================================================
# Contradiction detection
# ===================================================================

class TestContradictionDetection:

    def test_grade_2_with_negative_reasoning(self):
        c, _ = _detect_contradiction(2, "this book is not relevant at all")
        assert c is True

    def test_grade_0_with_positive_reasoning(self):
        c, _ = _detect_contradiction(0, "this is exactly what the user wants")
        assert c is True

    def test_consistent_grade_and_reasoning(self):
        c, _ = _detect_contradiction(2, "this is a great match for the query")
        assert c is False

    def test_grade_1_no_false_positive(self):
        c, _ = _detect_contradiction(1, "related but not a direct match")
        assert c is False


# ===================================================================
# _has_strict_majority
# ===================================================================

class TestHasStrictMajority:

    def test_clear_majority(self):
        has, w, cnt = _has_strict_majority([2, 2, 1])
        assert has and w == 2 and cnt == 2

    def test_two_way_tie_conservative(self):
        has, w, _ = _has_strict_majority([0, 0, 1, 1])
        assert has and w == 0, "2-way tie should resolve to lower grade"

    def test_three_way_split_no_majority(self):
        has, w, _ = _has_strict_majority([2, 1, 0])
        assert not has and w is None

    def test_escalated_two_way_tie_conservative(self):
        has, w, _ = _has_strict_majority([2, 1, 0, 2, 1])
        assert has and w == 1, "2-way tie between 1 and 2 -> conservative 1"

    def test_escalated_clear_majority(self):
        has, w, cnt = _has_strict_majority([2, 1, 0, 1, 1])
        assert has and w == 1 and cnt == 3

    def test_empty_list(self):
        has, w, _ = _has_strict_majority([])
        assert not has and w is None


# ===================================================================
# FIX 1 -- three-way split escalation
# ===================================================================

class TestThreeWaySplitEscalation:
    """Verify that a 3-way split [e.g. 2,1,0] escalates rather than
    silently becoming grade 0 via conservative tie-break."""

    def test_escalation_resolves(self, dry_run_client):
        """With max_escalation=2 the initial 3-way split draws 2 extra
        samples (5 total). With 3 bins and 5 items, pigeonhole guarantees
        at least a 2-way tie, so it always resolves."""
        split_q = _find_three_way_split_query()
        doc = {"work_id": "OL_SPLIT", "title": "Test Split Book"}

        jr = judge_pair(
            dry_run_client, split_q, doc,
            k=3, max_escalation=2, dry_run=True, rate_limit_sleep=0,
        )

        assert jr.grade is not None, "escalation must resolve (not None)"
        assert jr.k_requested == 5, "k_requested should be 3 + 2 escalation"
        assert jr.n_samples_ok == 5, "all dry-run samples should succeed"

        counts = Counter(jr.samples)
        win = counts[jr.grade]
        cands = [g for g, c in counts.items() if c == win]
        assert win * 2 > len(jr.samples) or len(cands) == 2, \
            f"grade {jr.grade} must have strict majority or be a 2-way tie"

    def test_no_consensus_returns_none_not_zero(self, dry_run_client):
        """With escalation disabled, a 3-way split must yield grade=None
        and reasoning='no_consensus', never grade 0."""
        split_q = _find_three_way_split_query()
        doc = {"work_id": "OL_SPLIT", "title": "Test Split Book"}

        jr = judge_pair(
            dry_run_client, split_q, doc,
            k=3, max_escalation=0, dry_run=True, rate_limit_sleep=0,
        )

        assert jr.grade is None, "grade must be None, not 0"
        assert jr.reasoning == "no_consensus"
        assert jr.low_confidence is True
        assert jr.k_requested == 3

    def test_no_consensus_is_unjudged_not_irrelevant(self, dry_run_client):
        """Downstream consumers must treat grade=None as unjudged. Verify
        that grade_distribution counts it under 'unjudged', not grade-0."""
        split_q = _find_three_way_split_query()
        doc = {"work_id": "OL_SPLIT", "title": "Test Split Book"}

        jr = judge_pair(
            dry_run_client, split_q, doc,
            k=3, max_escalation=0, dry_run=True, rate_limit_sleep=0,
        )
        # Include at least one graded pair so grade_distribution doesn't
        # short-circuit on "no valid grades found"
        graded = {"grade": 1, "samples": [1, 1, 1], "reasoning": "ok",
                  "low_confidence": False, "k_requested": 3, "n_samples_ok": 3}
        results = {"q": [asdict(jr), graded]}
        stats = grade_distribution(results)

        assert stats["unjudged"] == 1
        assert stats["no_consensus"] == 1
        assert stats["distribution"].get(0, 0) == 0, \
            "no_consensus must not inflate grade-0 count"


# ===================================================================
# FIX 2 -- agreement denominator is k_requested
# ===================================================================

class TestAgreementDenominator:
    """agreement must be win_count / k_requested, not win_count / n_samples_ok,
    so a single surviving sample from k=3 reads 1/3 instead of 1/1."""

    def test_k1_low_confidence(self, dry_run_client):
        jr = judge_pair(
            dry_run_client, "test_k1", {"work_id": "OL_K1", "title": "K1"},
            k=1, max_escalation=0, dry_run=True, rate_limit_sleep=0,
        )
        assert jr.n_samples_ok == 1
        assert jr.k_requested == 1
        assert jr.low_confidence is True, "k=1 must be low_confidence (< 2 ok samples)"
        assert jr.agreement == 1.0  # 1/1

    def test_k3_agreement_is_win_over_k(self, dry_run_client):
        jr = judge_pair(
            dry_run_client, "test_k3", {"work_id": "OL_K3", "title": "K3"},
            k=3, max_escalation=2, dry_run=True, rate_limit_sleep=0,
        )
        assert jr.n_samples_ok >= 3
        assert jr.low_confidence is False

        counts = Counter(jr.samples)
        win_count = max(counts.values())
        expected_agr = win_count / jr.k_requested
        assert jr.agreement == pytest.approx(expected_agr, abs=1e-3), \
            f"agreement should be win_count/k_requested = {expected_agr}"


# ===================================================================
# Grade distribution reporting
# ===================================================================

class TestGradeDistribution:

    MIXED = {
        "q1": [
            {"grade": 0, "samples": [0, 0, 0], "reasoning": "irrelevant",
             "low_confidence": False, "k_requested": 3, "n_samples_ok": 3},
            {"grade": 1, "samples": [1, 1, 0], "reasoning": "partial",
             "low_confidence": False, "k_requested": 3, "n_samples_ok": 3},
            {"grade": None, "samples": [2, 1, 0, 2, 1], "reasoning": "no_consensus",
             "low_confidence": True, "k_requested": 5, "n_samples_ok": 5},
        ],
        "q2": [
            {"grade": 2, "samples": [2, 2, 2], "reasoning": "perfect",
             "low_confidence": False, "k_requested": 3, "n_samples_ok": 3},
        ],
    }

    def test_no_consensus_counted(self):
        stats = grade_distribution(self.MIXED)
        assert stats["no_consensus"] == 1

    def test_low_confidence_counted(self):
        stats = grade_distribution(self.MIXED)
        assert stats["low_confidence"] == 1

    def test_unjudged_counted(self):
        stats = grade_distribution(self.MIXED)
        assert stats["unjudged"] == 1

    def test_permissive_judge_low_negative_rate(self):
        """Negative rate below 20 % must be flagged."""
        permissive = {
            "q1": [{"grade": 1, "samples": [1]}, {"grade": 2, "samples": [2]}],
            "q2": [{"grade": 2, "samples": [2]}, {"grade": 1, "samples": [1]},
                   {"grade": 2, "samples": [2]}],
        }
        stats = grade_distribution(permissive)
        assert stats["negative_rate"] < 0.20

    def test_healthy_distribution_passes(self):
        healthy = {
            "q1": [{"grade": 0, "samples": [0]}, {"grade": 0, "samples": [0]},
                   {"grade": 1, "samples": [1]}],
            "q2": [{"grade": 2, "samples": [2]}, {"grade": 0, "samples": [0]}],
        }
        stats = grade_distribution(healthy)
        assert stats["negative_rate"] >= 0.20


# ===================================================================
# Self-consistency report
# ===================================================================

class TestSelfConsistencyReport:

    MIXED = TestGradeDistribution.MIXED

    def test_covers_multi_sample_pairs(self):
        sc = self_consistency_report(self.MIXED)
        assert sc["total_pairs"] >= 3

    def test_includes_low_confidence_count(self):
        sc = self_consistency_report(self.MIXED)
        assert "low_confidence_pairs" in sc

    def test_includes_partial_failure_count(self):
        sc = self_consistency_report(self.MIXED)
        assert "partial_failure_pairs" in sc


# ===================================================================
# CSV export
# ===================================================================

class TestCSVExport:

    MIXED = TestGradeDistribution.MIXED

    def test_file_created(self, tmp_path):
        p = tmp_path / "audit.csv"
        export_audit_csv(self.MIXED, p)
        assert p.exists()

    def test_row_count(self, tmp_path):
        p = tmp_path / "audit.csv"
        export_audit_csv(self.MIXED, p)
        with open(p, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4

    def test_no_consensus_sorted_first(self, tmp_path):
        p = tmp_path / "audit.csv"
        export_audit_csv(self.MIXED, p)
        with open(p, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["grade"] == "", "no_consensus row must be first"
        assert rows[0]["low_confidence"] == "True"

    def test_has_new_columns(self, tmp_path):
        p = tmp_path / "audit.csv"
        export_audit_csv(self.MIXED, p)
        with open(p, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for col in ("k_requested", "n_samples_ok", "low_confidence"):
            assert col in rows[0], f"missing column: {col}"
