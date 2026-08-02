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

# Sub-batch size for cross-encoder inference. Candidates are sorted by token
# length and scored in groups of this size, each padded only to its own longest
# member rather than to the longest in the whole candidate set.
#
# Padding is masked by attention_mask, so this does not change any score — it
# only stops the model from doing arithmetic on pad tokens. Verified: max
# absolute score difference vs. a single flat batch is exactly 0.0 across 12
# trials of 25 real candidates, with zero ranking changes.
#
# Measured sweep (25 candidates, local CPU, vs. one flat batch at 1554 ms):
#     bs=2  676 ms  2.30x     bs=8   870 ms  1.79x
#     bs=4  730 ms  2.13x     bs=16 1106 ms  1.41x
#     bs=6  781 ms  1.99x     bs=25 1587 ms  0.98x
#
# The curve is flat between 2 and 6, so 4 is chosen over the nominally faster
# 2: it captures nearly all the win with half the ONNX Runtime invocations,
# which is the cost most likely to grow on the smaller production container.
# bs=25 reproducing the flat-batch time is the sanity check that the bucketing
# is doing what it claims — one bucket spanning everything is the old behaviour.
RERANK_BATCH_SIZE = 4


def rerank_fetch_k(top_k: int) -> int:
    """Candidate count retrieval should fetch to feed the reranker."""
    return max(top_k, int(top_k * RERANK_DEPTH_MULTIPLIER))
