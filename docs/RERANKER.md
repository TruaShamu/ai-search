# Reranker Design

How and why BookSearch uses a cross-encoder reranker as an optional second stage,
the model choice and its limitations, and the performance engineering that made
it viable on CPU.

---

## Why Two-Stage Retrieval

First-stage retrieval (TF-IDF + dense via RRF) is fast (~216 ms) because it
scores documents independently — each vector comparison is O(1) and Qdrant runs
them in parallel. But independent scoring cannot model the *interaction* between
a query and a document: "a heist that goes wrong" and a description of *Freezer
Burn* share no tokens, and their relationship only emerges from reading both
together.

A cross-encoder does exactly that — it concatenates `[query, passage]` and runs
them through a transformer jointly, producing a relevance score that conditions
on both. This is dramatically more expensive (one forward pass *per candidate*
instead of one lookup), which is why it can only rescore a small shortlist rather
than the full 84,801-document index.

The tradeoff is explicit: retrieve 25 candidates cheaply (2.5× the returned
top-k), then rerank them expensively. Measured payoff: **+0.081 NDCG over
hybrid**, with **zero regressions across 80 paired known-item queries**
([results](EVAL_RESULTS.md#reranking-on-v2-a-paired-test)).

---

## Model: ms-marco-MiniLM-L-6-v2

A 22M-parameter cross-encoder from the `cross-encoder/ms-marco-*` family,
chosen for CPU viability:

| Model | Layers | Params | MSMARCO MRR@10 | CPU inference (25 docs) |
|---|---|---|---|---|
| ms-marco-MiniLM-L-6-v2 | 6 | 22 M | ~36 | **~1.8 s (ONNX)** |
| ms-marco-MiniLM-L-12-v2 | 12 | 33 M | ~38 | ~3.5 s (estimated) |
| ms-marco-electra-base | 12 | 110 M | ~39 | ~7 s (estimated) |

The L-6 model is the smallest that produces measurable gains. On a CPU-only
deployment where reranking already costs ~10× the retrieval latency, doubling
model size for ~2 MSMARCO points is the wrong trade.

### Limitations

**Domain mismatch.** The model was trained on MS MARCO — Bing web search
passages, not book descriptions. It has never seen book metadata during training.
`src/reranker/passage.py` works around this by constructing natural prose
(`"Title by Author. Description. This book covers X, Y, Z."`) rather than
pipe-delimited structured fields, because the model expects web-like text. This
is domain adaptation without fine-tuning, and there is no measurement of how
much quality the mismatch costs. Fine-tuning on book relevance data is the
obvious next step, but requires labeled training pairs the project does not
currently have.

**512 token limit.** The tokenizer truncates at 512 tokens
(`config.py:MAX_SEQUENCE_TOKENS`). `MAX_DESCRIPTION_CHARS = 1500` is sized so
the token limit is the binding constraint rather than the character cap — but the
two are coupled and easy to break independently. A `[:300]` character truncation
was already caught once discarding 60% of description text and making reranking
look harmful; the current 1500-char limit covers >99% of descriptions but the
coupling is not enforced by a test.

**English-only.** Aligns with the English-only index, but means the reranker
cannot rescue any foreign-language leakage that slips past the language filter.

**No fine-tuning.** Used entirely off-the-shelf. The eval shows it works, but
there is no measurement of what domain-specific fine-tuning would gain. The
zero-regression result across 80 paired queries suggests the model is not
overfitting to surface similarity, but n=80 is small.

---

## ONNX Optimization

The reranker is exported to ONNX and served via ONNX Runtime on CPU. Measured
speedup on the same 25-candidate input:

| Backend | Latency (25 docs) | Speedup |
|---|---|---|
| PyTorch | 86 ms (4 docs), ~3.6 s (25 docs) | — |
| ONNX Runtime | 23 ms (4 docs), ~1.8 s (25 docs) | **3.7×** (per-call) |

The ONNX model (86 MB) is stored in Azure Blob Storage, downloaded by the
deployment workflow, and baked into the API container image at
`data/models/reranker-onnx/`. Reranker initialization fails clearly if the
artifact is missing rather than silently switching to a slower backend.

---

## Length-Bucketed Batching

The single largest performance win after ONNX export. Scoring all 25 candidates
in one batch with `padding=True` lets a single long description set the tensor
width for every short one — measured at **45.2% wasted padding tokens**.

The fix: sort candidates by token length, score in sub-batches of 4, each padded
only to its own longest member. This cuts padding to 17.7%.

| | Flat batch | Length-bucketed | Change |
|---|---|---|---|
| Rerank stage | 3,642 ms | **1,812 ms** | **2.01× faster** |
| End-to-end | 4,096 ms | **2,089 ms** | 1.96× faster |
| Padding tokens | 45.2% | **17.7%** | −33.4% |

**This changes no score.** Padding is masked by `attention_mask`, so the model's
arithmetic on real tokens is identical. Verified: max absolute score difference
vs. a single flat batch is **exactly 0.0** across 12 trials of 25 real
candidates, with zero ranking changes.

**Batch size 4** came from a measured sweep:

| Batch size | Latency | Speedup vs flat |
|---|---|---|
| 2 | 676 ms | 2.30× |
| **4** | **730 ms** | **2.13×** |
| 6 | 781 ms | 1.99× |
| 8 | 870 ms | 1.79× |
| 25 (= flat) | 1,587 ms | 1.00× |

The curve is flat between 2 and 6. Batch size 4 captures nearly all the win with
half the ONNX Runtime invocations of 2 — the cost most likely to grow on a
smaller container.

**The riskiest part is the scatter-back.** After sorting by length and scoring in
sub-batches, scores must be mapped back to their original candidate positions.
Returning them in sorted order would silently mis-attribute every score while
still producing a plausible-looking ranking. `tests/test_reranker_batching.py`
covers this with fakes rather than the 86 MB ONNX model, so it runs on a clean
clone — and reverting the scatter to sequential assignment fails 8 of its 11
tests.

---

## Passage Construction

`src/reranker/passage.py:build_passage` constructs the text the cross-encoder
sees. The design is driven by the domain mismatch: ms-marco expects web prose,
not structured metadata.

```
{title} by {authors}. {description[:1500]}. This book covers {subjects}.
```

- **Natural prose, not structured fields.** Pipe-delimited or key-value formats
  are out-of-distribution for a model trained on web paragraphs.
- **Subject tags are cleaned.** `clean_subjects` splits comma-separated entries,
  drops machine-metadata tokens (`isbn:...`, `=2004`), deduplicates
  case-insensitively, and removes strict substrings (`Fiction` is dropped if
  `Historical Fiction` is present).
- **Description is capped at 1,500 chars.** Sized so the tokenizer's 512-token
  limit is the binding constraint. The coupling between these two numbers is
  documented in `config.py` but not enforced by a test.
