"""Tests for length-bucketed cross-encoder batching.

The reranker sorts candidates by token length, scores them in sub-batches
padded only to each bucket's own maximum, then scatters the scores back to
the caller's original order.

The scatter-back is the dangerous part: returning scores in *sorted* order
would silently mis-attribute every score to the wrong document while still
producing a plausible-looking ranking. These tests use fakes rather than the
real 86 MB ONNX model so they run on a clean clone with no downloaded data.
"""

import numpy as np
import pytest

import src.reranker.onnx_reranker as onnx_mod
from src.reranker.onnx_reranker import OnnxReranker


class _FakeInput:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    """Scores each row as its own first token id.

    Passage i is tokenized as a run of i's, so the expected score for passage
    i is exactly i. Any failure to undo the length sort therefore shows up as
    a permutation, not as a subtle numeric drift.
    """

    def __init__(self) -> None:
        self.batch_widths: list[int] = []

    def get_inputs(self) -> list[_FakeInput]:
        return [_FakeInput("input_ids"), _FakeInput("attention_mask")]

    def run(self, _outputs, feed):
        ids = feed["input_ids"]
        self.batch_widths.append(int(ids.shape[1]))
        return [ids[:, 0].astype(np.float32).reshape(-1, 1)]


class _FakeTokenizer:
    """Turns passage "i" into a token run of value i with length lengths[i]."""

    pad_token_id = 0

    def __init__(self, lengths: list[int]) -> None:
        self.lengths = lengths

    def __call__(self, _queries, passages, **_kwargs):
        input_ids, attention_mask = [], []
        for p in passages:
            i = int(p)
            n = self.lengths[i]
            input_ids.append([i] * n)
            attention_mask.append([1] * n)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


# Deliberately jumbled so sorting genuinely reorders the batch.
LENGTHS = [7, 3, 19, 1, 5, 12, 2, 25, 4, 9]


def _make_reranker(lengths: list[int]) -> OnnxReranker:
    """Build a reranker without running __init__ (which loads the ONNX model)."""
    rr = OnnxReranker.__new__(OnnxReranker)
    rr.backend = "onnx"
    rr.tokenizer = _FakeTokenizer(lengths)
    rr.session = _FakeSession()
    return rr


class TestBucketedBatching:

    def test_scores_map_back_to_original_order(self, monkeypatch):
        monkeypatch.setattr(onnx_mod, "RERANK_BATCH_SIZE", 4)
        rr = _make_reranker(LENGTHS)
        passages = [str(i) for i in range(len(LENGTHS))]

        scores = rr._predict_onnx("q", passages)

        # Passage i must score exactly i despite being processed out of order.
        assert scores.tolist() == [float(i) for i in range(len(LENGTHS))]

    def test_buckets_are_narrower_than_a_flat_batch(self, monkeypatch):
        monkeypatch.setattr(onnx_mod, "RERANK_BATCH_SIZE", 4)
        rr = _make_reranker(LENGTHS)

        rr._predict_onnx("q", [str(i) for i in range(len(LENGTHS))])

        widths = rr.session.batch_widths
        assert len(widths) == 3, "10 passages at batch size 4 should be 3 calls"
        # A flat batch would pad everything to the global max (25).
        assert max(widths) == max(LENGTHS)
        assert sum(w < max(LENGTHS) for w in widths) >= 2
        # Bucketing must strictly reduce total padded token volume.
        bucketed = sum(w * n for w, n in zip(widths, (4, 4, 2)))
        assert bucketed < max(LENGTHS) * len(LENGTHS)

    def test_batch_size_covering_everything_matches_one_flat_batch(self, monkeypatch):
        """bs >= n degenerates to the old single-batch behaviour."""
        monkeypatch.setattr(onnx_mod, "RERANK_BATCH_SIZE", 100)
        rr = _make_reranker(LENGTHS)

        scores = rr._predict_onnx("q", [str(i) for i in range(len(LENGTHS))])

        assert rr.session.batch_widths == [max(LENGTHS)]
        assert scores.tolist() == [float(i) for i in range(len(LENGTHS))]

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 7, 100])
    def test_order_preserved_across_batch_sizes(self, monkeypatch, batch_size):
        monkeypatch.setattr(onnx_mod, "RERANK_BATCH_SIZE", batch_size)
        rr = _make_reranker(LENGTHS)

        scores = rr._predict_onnx("q", [str(i) for i in range(len(LENGTHS))])

        assert scores.tolist() == [float(i) for i in range(len(LENGTHS))]

    def test_empty_candidate_list(self, monkeypatch):
        monkeypatch.setattr(onnx_mod, "RERANK_BATCH_SIZE", 4)
        rr = _make_reranker(LENGTHS)

        scores = rr._predict_onnx("q", [])

        assert scores.shape == (0,)
        assert rr.session.batch_widths == []

    def test_padding_uses_tokenizer_pad_id(self, monkeypatch):
        """Pad positions must not collide with a real token id."""
        monkeypatch.setattr(onnx_mod, "RERANK_BATCH_SIZE", 2)
        lengths = [1, 6]
        rr = _make_reranker(lengths)
        captured = {}

        original_run = rr.session.run

        def spy(outputs, feed):
            captured["input_ids"] = feed["input_ids"].copy()
            captured["attention_mask"] = feed["attention_mask"].copy()
            return original_run(outputs, feed)

        rr.session.run = spy
        rr._predict_onnx("q", ["0", "1"])

        ids = captured["input_ids"]
        mask = captured["attention_mask"]
        # Row for passage 0 is length 1, padded out to width 6.
        assert ids.shape == (2, 6)
        assert mask[0].tolist() == [1, 0, 0, 0, 0, 0]
        assert ids[0].tolist()[1:] == [_FakeTokenizer.pad_token_id] * 5
