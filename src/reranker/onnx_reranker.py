"""ONNX-optimized cross-encoder reranker.

2-3x faster than PyTorch on CPU by using ONNX Runtime.
Falls back to PyTorch if ONNX model not found.

Usage:
    reranker = OnnxReranker()  # auto-detects ONNX or PyTorch
    result = reranker.rerank("query", candidates, top_k=10)
"""

import re
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

ONNX_DIR = Path("data/models/reranker-onnx")
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_MACHINE_METADATA_RE = re.compile(r"^\w+:\S+=|=\d{4}\b")
_WHITESPACE_RE = re.compile(r"\s+")


def _clean_subjects(raw: list[str], max_subjects: int = 5) -> list[str]:
    """Clean and deduplicate subject tags for natural prose rendering.

    Splits comma-separated entries, drops machine-metadata tokens,
    deduplicates case-insensitively, and removes strict substrings.
    """
    # Flatten comma-separated multi-topic entries
    flat = []
    for entry in raw:
        flat.extend(part.strip() for part in entry.split(",") if part.strip())

    # Drop machine-metadata tokens (e.g. "Nyt:Mass-Market-Monthly=2021-11-07")
    filtered = [s for s in flat if not _MACHINE_METADATA_RE.search(s)]

    # Case-insensitive dedupe (normalizing hyphens), preserving first-seen order
    seen: set[str] = set()
    deduped: list[str] = []
    for s in filtered:
        key = s.lower().replace("-", " ")
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    # Drop entries that are a strict substring of any other kept entry
    result = []
    for s in deduped:
        s_lower = s.lower()
        if not any(s_lower != o.lower() and s_lower in o.lower() for o in deduped):
            result.append(s)

    return result[:max_subjects]


class OnnxReranker:
    """Cross-encoder reranker with ONNX Runtime backend."""

    def __init__(self, onnx_dir: Path = ONNX_DIR):
        self.onnx_dir = onnx_dir
        onnx_path = onnx_dir / "model.onnx"

        if onnx_path.exists():
            import onnxruntime as ort

            self.backend = "onnx"
            self.tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))
            self.session = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )
            print(f"Reranker loaded (ONNX): {onnx_dir}")
        else:
            # Fallback to PyTorch
            from src.reranker.model import CrossEncoderReranker

            self.backend = "pytorch"
            self._pytorch_reranker = CrossEncoderReranker()
            print("Reranker loaded (PyTorch fallback — run onnx_export for speedup)")

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

    def _build_passage(self, doc: dict) -> str:
        """Build passage text from document as natural prose for cross-encoder.

        ms-marco was trained on web prose, so we avoid pipe-delimited metadata.
        The tokenizer's MAX_SEQUENCE_TOKENS limit handles final truncation;
        MAX_DESCRIPTION_CHARS is a cheap guard sized to keep the token limit as
        the binding constraint.  These two constants are coupled — see config.py.
        """
        parts = []
        title = doc.get("title", "")
        authors = doc.get("authors", "")
        if title and authors:
            parts.append(f"{title} by {authors}.")
        elif title:
            parts.append(f"{title}.")

        if doc.get("description"):
            # Strip stray quote wrapping and collapse whitespace (\r\n, tabs, etc.)
            desc = doc["description"].strip("\"'")
            desc = _WHITESPACE_RE.sub(" ", desc).strip()
            if desc:
                parts.append(desc[:MAX_DESCRIPTION_CHARS])

        if doc.get("subjects"):
            subjects = doc["subjects"]
            if isinstance(subjects, list):
                cleaned = _clean_subjects(subjects)
                if cleaned:
                    parts.append(f"This book covers {', '.join(cleaned)}.")

        return " ".join(parts)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 10) -> dict:
        """Rerank candidates. Uses ONNX if available, else PyTorch."""
        if self.backend == "pytorch":
            return self._pytorch_reranker.rerank(query, candidates, top_k)

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


if __name__ == "__main__":
    # Quick test
    reranker = OnnxReranker()

    query = "romance set in Scotland"
    candidates = [
        {"title": "Desmond goes to Scotland", "authors": "Althea", "description": "A children's picture book about a bear visiting Scotland.", "subjects": ["Children's fiction"], "score": 24.7},
        {"title": "Seducing the Highlander", "authors": "Emma Wildes", "description": "Three stories of romance, adventure, and passion in the Scottish Highlands.", "subjects": ["Fiction, Romance, Historical"], "score": 0.80},
        {"title": "Computational Logic and Set Theory", "authors": "Jacob Schwartz", "description": "A technical book about mathematical logic.", "subjects": ["Mathematics"], "score": 19.5},
        {"title": "The Bride", "authors": "Julie Garwood", "description": "A Scottish laird must take an English bride. A feisty beauty. Passion in the Highlands.", "subjects": ["Fiction, Romance, Historical", "Scotland In Fiction"], "score": 0.78},
    ]

    print(f"Backend: {reranker.backend}")
    print(f"\nQuery: \"{query}\"")

    result = reranker.rerank(query, candidates, top_k=4)
    print(f"Latency: {result['latency_ms']}ms")
    print("\nReranked results:")
    for r in result["results"]:
        change = r["rank_change"]
        arrow = f"+{change}" if change > 0 else str(change) if change < 0 else "="
        print(f"  {r['rerank_score']:+.4f} | {r['title']} (was #{r['original_rank']}, {arrow})")
