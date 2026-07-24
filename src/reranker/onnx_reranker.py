"""ONNX-optimized cross-encoder reranker.

2-3x faster than PyTorch on CPU by using ONNX Runtime.
Falls back to PyTorch if ONNX model not found.

Usage:
    reranker = OnnxReranker()  # auto-detects ONNX or PyTorch
    result = reranker.rerank("query", candidates, top_k=10)
"""

import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ONNX_DIR = Path("data/models/reranker-onnx")
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


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
        """Score (query, passage) pairs using ONNX Runtime."""
        encoded = self.tokenizer(
            [query] * len(passages),
            passages,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )

        ort_inputs = {k: v for k, v in encoded.items() if k in [n.name for n in self.session.get_inputs()]}
        outputs = self.session.run(None, ort_inputs)
        # Output shape: (batch, 1) — squeeze to (batch,)
        scores = outputs[0].squeeze(-1) if outputs[0].ndim > 1 else outputs[0]
        return scores

    def _build_passage(self, doc: dict) -> str:
        """Build passage text from document."""
        parts = []
        if doc.get("title"):
            parts.append(doc["title"])
        if doc.get("authors"):
            parts.append(f"by {doc['authors']}")
        if doc.get("description"):
            parts.append(doc["description"][:300])
        if doc.get("subjects"):
            subjects = doc["subjects"]
            if isinstance(subjects, list):
                parts.append(f"Subjects: {', '.join(subjects[:5])}")
        return " | ".join(parts)

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
    print(f"\nReranked results:")
    for r in result["results"]:
        change = r["rank_change"]
        arrow = f"+{change}" if change > 0 else str(change) if change < 0 else "="
        print(f"  {r['rerank_score']:+.4f} | {r['title']} (was #{r['original_rank']}, {arrow})")
