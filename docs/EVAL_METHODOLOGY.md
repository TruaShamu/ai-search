# Evaluation Methodology

How BookSearch's retrieval quality is measured, what the measurements can and
cannot support, and how to reproduce them. For the actual numbers, see
[EVAL_RESULTS.md](EVAL_RESULTS.md).

Corpus background and the superseded v1 results live in
[CORPUS_HISTORY.md](CORPUS_HISTORY.md).

---

## Two Independent Harnesses

Quality is measured two ways on purpose, because each covers the other's blind
spot.

| | Graded relevance eval | Known-item eval |
|---|---|---|
| Script | `scripts/graded_eval.py` | `python -m src.eval.known_item_eval` |
| Question | *Given an exploratory query, how good is the ranking?* | *Given an exact title, is that book #1?* |
| Labels | LLM judge, graded 0/1/2 | Objective — the sampled book either comes back first or it does not |
| Strength | Measures the actual product | No judge, no pooling, no subjectivity; verifiable by hand |
| Weakness | Depends on a judge that agrees with a human ~2/3 of the time | Easiest possible task; not representative of real queries |

A claim is only treated as solid here when both agree, or when the one that
applies is explicitly named.

---

## How Queries Are Built

**Queries are grounded in the corpus.** 100 queries are generated *from sampled
corpus documents*, so each carries known gold document ids. An earlier version of
this eval generated queries from an LLM with no view of the index, which produced
canonical titles (*1984*, *The Great Gatsby*) against a corpus of mostly obscure
works — every mode scored near zero and the differences were noise. Generation
now validates every gold id against the live index before proceeding.

**Candidates are pooled across modes.** For each query, the top results from
keyword, vector, hybrid, and hybrid+rerank are unioned into a pool of up to 50
documents, so no mode is judged on candidates only its rivals surfaced.

**Judging is graded, not binary.** Each of the 5,000 query–document pairs is
scored 0 (irrelevant) / 1 (partially relevant) / 2 (fully relevant) by
`gpt-5.4-nano`. Zeros are kept rather than discarded — a mode that returns
judged-irrelevant documents must be penalized for it. Failures are recorded as
`null` and excluded rather than silently counted as 0.

Grade distribution across the 5,000 pairs: **2,720 zeros (54.4%), 1,684 ones
(33.7%), 596 twos (11.9%)**. This run had **0 unjudged pairs**.

**Comparisons are paired.** Mode differences use a bootstrap over per-query
*deltas* rather than a comparison of independent means. Query difficulty varies
far more than the gap between modes, so pairing is what makes these differences
resolvable at n=98.

**Runs abort rather than truncate.** Search requests retry with backoff, and the
harness refuses to write results if any query is missing from any mode. This
exists because a dropped connection once produced a complete-looking
`results.json` computed over 43 of 98 queries — plausible numbers, wrong sample.

---

## Determinism

Hybrid search was **non-deterministic** until it was measured and fixed.

Qdrant's server-side RRF assigns each document `sum(1/(k + rank))` over the lists
it appears in, so two documents at the same rank in exactly one input list each
receive **bit-identical scores**. Ties are common in fusion — unlike raw dense
scores, which are floats that effectively never collide — and Qdrant breaks them
by segment-merge order, which is not stable across identical requests. Because
the server also truncated at exactly the requested limit, a tie straddling that
boundary dropped a different document on every call.

Measured over 40 queries, two back-to-back identical hybrid requests returned:

| | Before | After |
|---|---|---|
| Different result order | 28 / 40 | **0 / 40** |
| Different result set | 11 / 40 | **0 / 40** |
| Different rank-1 document | 8 / 40 | **0 / 40** |

Keyword and vector were stable in all 40 both before and after, as expected.

This was also the source of the ~2pp run-to-run drift previously observed in the
known-item benchmark, and it was found by noticing that two independent eval runs
agreed exactly on keyword, vector and rerank MRR but disagreed on hybrid.

The fix fetches a fixed margin past the window it returns, then re-sorts with an
explicit `(-score, work_id)` tie-break and truncates client-side. Prefetch limits
stay keyed to the requested depth, so the fused ranking itself is unchanged — the
client simply stops letting the server make the cut. Cost in latency: none
measurable. Covered by `tests/test_tie_break.py`.

Two independent CI runs now reproduce every metric in this document to four
decimals.

---

## Judge Validation and Known Limitations

Stated plainly, because they bound what the [results](EVAL_RESULTS.md) can
support.

- **The judge has been validated against human labels, and the agreement is only
  fair.** 89 pairs were hand-labeled across two rounds with the machine's grade
  hidden (`python -m src.eval.label`). On the **representative random sample
  (n=40)** — the round whose kappa is directly interpretable — Cohen's kappa is
  **0.314 [0.056, 0.562]** at 65% raw agreement. A separate stratified round
  (n=49) scores 0.542, but stratified sampling over-represents rare grades and
  inflates kappa; post-stratifying back to true prevalence gives **0.308**,
  independently reproducing the random round's 0.314. That is **"fair" agreement,
  not "substantial"** — every graded number rests on labels a human agrees with
  about two-thirds of the time.
- **That validation was performed on v1 pairs and has not been repeated on v2.**
  The judge prompt and model are unchanged, but the corpus it reads is not, so
  the kappa above is evidence about the judge in general rather than about this
  specific run.
- **The judge errs strict, not lenient.** Across all 89 human labels, the human
  graded *higher* than the judge 24 times and *lower* 5 times. Whatever else the
  labels get wrong, they are more likely to understate this system than flatter
  it.
- **Query generation and judging share a model family,** so blind spots may be
  correlated rather than independent. The specific concern was that a
  cross-encoder and an LLM judge both reward surface semantic similarity, which
  would inflate the measured reranking gain. The human labels argue against it:
  of the 20 pairs the judge graded 2, the human called **zero** irrelevant (18
  exact agreements, 2 downgraded to "partial"). n=20 is small enough that this
  weakens the worry rather than closing it.
- **Per-category results are underpowered.** Four of five categories have n < 30.
  `author` (n=13) and `combined` (n=9) have intervals that include zero.
- **Recall is capped by construction** at a mean ceiling of 0.559, so recall is
  not comparable across systems — or across corpus versions.
- **Documents outside the judged pool count as non-relevant.** Standard for
  pooled evaluation. Verified not to penalize reranking: across sampled queries,
  0 of 120 reranked top-10 documents fell outside the pool.
- **Two queries were dropped** for having no relevant document, which slightly
  biases those categories toward queries the system could already answer.
- **Labels come from a zero-shot `gpt-5.4-nano` judge**, not the richer
  calibrated judge in `src/eval/judge.py` — `scripts/graded_eval.py` uses its
  own simpler prompt.

Reproduce the agreement analysis:

```bash
python -m src.eval.judge --agreement \
  data/eval/v2/judgments.v1-openlibrary.json \
  data/eval/v2/human_random.v1-openlibrary.json
```

---

## Reproducing

Everything below the judging step is free; judging 5,000 pairs costs Azure OpenAI
tokens.

```bash
# Full pipeline: generate -> pool -> judge -> eval
python scripts/graded_eval.py --step all

# Individual steps
python scripts/graded_eval.py --step generate   # corpus-grounded queries
python scripts/graded_eval.py --step pool       # union candidates across modes
python scripts/graded_eval.py --step judge      # LLM grading (costs tokens)
python scripts/graded_eval.py --step eval       # metrics, no judging cost

# Threshold sensitivity (offline: reuses committed ranked lists)
python scripts/threshold_sensitivity.py --offline

# Known-item eval against the live deployment
python -m src.eval.known_item_eval --modes hybrid,hybrid+rerank
```

### Running it remotely

The pipeline also runs as a GitHub Actions workflow
(`.github/workflows/eval.yml`, `workflow_dispatch`), which is the preferred way
to produce publishable numbers:

```bash
gh workflow run eval.yml -f step=eval -f threshold_sensitivity=true
```

Two reasons it lives in CI. **Reproducibility** — results come from a declared
environment anyone can inspect rather than one laptop. **Reliability** — a home
network dropping mid-run is not hypothetical; it is what produced the 43-of-98
partial run that motivated the abort-on-failure guard.

The job installs only `httpx`, `python-dotenv`, and `numpy`: the scripts talk to
the deployed API over HTTP, so none of torch/transformers/onnxruntime is needed.
All eval inputs — queries, judgments, and the pooled candidate set — are
committed, so a runner can reproduce the metrics without re-judging.

### Committed evidence

| File | What it is |
|---|---|
| `data/eval/v2/queries_grounded.json` | The 100 corpus-grounded queries with gold ids |
| `data/eval/v2/pooled.json` | Pooled candidates per query (the judging input) |
| `data/eval/v2/judgments.json` | All 5,000 graded pairs |
| `data/eval/v2/results.json` | Computed metrics with confidence intervals |
| `data/eval/v2/rankings.json` | Raw per-mode ranked lists |
| `data/eval/v2/threshold_sensitivity.json` | Lenient vs strict threshold report |
| `data/eval/v2/*.v1-openlibrary.json` | Human labels and judge output from the v1 run, kept so the agreement analysis stays reproducible |
