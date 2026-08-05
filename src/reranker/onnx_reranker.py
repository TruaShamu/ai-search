"""ONNX-optimized cross-encoder reranker.

2-3x faster than PyTorch on CPU by using ONNX Runtime.

Usage:
    reranker = OnnxReranker()
    result = reranker.rerank("query", candidates, top_k=10)
"""

import logging
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from src.reranker.config import (  # noqa: F401  (re-exported for callers)
    MAX_DESCRIPTION_CHARS,
    MAX_SEQUENCE_TOKENS,
    RERANK_BATCH_SIZE,
    RERANK_DEPTH_MULTIPLIER,
)
from src.reranker.passage import build_passage

logger = logging.getLogger(__name__)

ONNX_DIR = Path("data/models/reranker-onnx")


class OnnxReranker:
    """Cross-encoder reranker with ONNX Runtime backend."""

    def __init__(self, onnx_dir: Path = ONNX_DIR):
        self.onnx_dir = onnx_dir
        onnx_path = onnx_dir / "model.onnx"

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"Missing ONNX reranker model at {onnx_path}. "
                "Provision data/models/reranker-onnx before enabling reranking."
            )

        import onnxruntime as ort

        self.backend = "onnx"
        self.tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        logger.info("Reranker loaded (ONNX): %s", onnx_dir)

    def _predict_onnx(self, query: str, passages: list[str]) -> np.ndarray:
        """Score (query, passage) pairs using ONNX Runtime.

        Passages are sorted by token length and scored in length-bucketed
        sub-batches, each padded only to its own longest member.

        This does not change any score. Padding is masked out by
        attention_mask, so the arithmetic on real tokens is identical; the
        only difference is how many pad tokens the model multiplies by zero.
        Scoring one flat batch let a single long description set the tensor
        width for every short one -- measured at ~47% wasted tokens on a
        25-candidate set.
        """
        if not passages:
            return np.empty(0, dtype=np.float32)

        encoded = self.tokenizer(
            [query] * len(passages),
            passages,
            padding=False,
            truncation=True,
            max_length=MAX_SEQUENCE_TOKENS,
        )

        input_names = {n.name for n in self.session.get_inputs()}
        keys = [k for k in encoded.keys() if k in input_names]
        lengths = [len(ids) for ids in encoded["input_ids"]]

        # Pad with the tokenizer's own pad id; attention_mask marks these
        # positions dead, so the value only matters for embedding-table bounds.
        pad_id = self.tokenizer.pad_token_id or 0

        order = sorted(range(len(passages)), key=lambda i: lengths[i])
        scores = np.empty(len(passages), dtype=np.float32)

        for start in range(0, len(order), RERANK_BATCH_SIZE):
            idx = order[start:start + RERANK_BATCH_SIZE]
            width = max(lengths[i] for i in idx)

            batch = {}
            for key in keys:
                fill = pad_id if key == "input_ids" else 0
                batch[key] = np.array(
                    [
                        encoded[key][i] + [fill] * (width - lengths[i])
                        for i in idx
                    ],
                    dtype=np.int64,
                )

            out = self.session.run(None, batch)[0]
            out = out.squeeze(-1) if out.ndim > 1 else out
            # Scatter back: idx holds original positions, so this undoes the
            # length sort rather than returning scores in sorted order.
            scores[idx] = out.astype(np.float32)

        return scores

    @staticmethod
    def _build_passage(doc: dict) -> str:
        return build_passage(doc)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 10) -> dict:
        """Rerank candidates with ONNX Runtime."""
        if not candidates:
            return {"results": [], "latency_ms": 0, "candidates_scored": 0}

        start = time.time()

        # Build passages
        passages = [self._build_passage(doc) for doc in candidates]

        # Score with ONNX
        scores = self._predict_onnx(query, passages)

        # Sort by score
        scored = list(zip(candidates, scores, range(1, len(candidates) + 1)))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Build results
        results = []
        for new_rank, (doc, score, original_rank) in enumerate(scored[:top_k], 1):
            result = dict(doc)
            result["rerank_score"] = round(float(score), 4)
            result["original_rank"] = original_rank
            result["rank_change"] = original_rank - new_rank
            results.append(result)

        latency_ms = (time.time() - start) * 1000

        return {
            "results": results,
            "latency_ms": round(latency_ms, 1),
            "candidates_scored": len(candidates),
            "backend": self.backend,
        }
