"""Reranker tuning constants.

Deliberately dependency-free so lightweight callers (eval scripts, CI gates) can
import these numbers without pulling in transformers/torch/onnxruntime.
"""

# How many extra candidates retrieval fetches when reranking is enabled:
#   fetch_k = int(top_k * RERANK_DEPTH_MULTIPLIER)
#
# Kept modest because passages are no longer truncated to 300 chars, so each
# candidate costs meaningfully more to score. Any evaluation of reranking
# headroom must use this same depth, or it measures a system that isn't shipped.
RERANK_DEPTH_MULTIPLIER = 2.5

# Max raw description characters fed into a passage. Sized so the tokenizer's
# MAX_SEQUENCE_TOKENS limit stays the binding constraint rather than this cap.
# These two numbers are coupled — changing one without the other silently
# re-introduces arbitrary truncation.
MAX_DESCRIPTION_CHARS = 1500
MAX_SEQUENCE_TOKENS = 512


def rerank_fetch_k(top_k: int) -> int:
    """Candidate count retrieval should fetch to feed the reranker."""
    return max(top_k, int(top_k * RERANK_DEPTH_MULTIPLIER))
