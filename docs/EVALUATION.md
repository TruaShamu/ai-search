# Evaluation Methodology and Full Results

Complete methodology, results, and limitations for BookSearch's retrieval
evaluation. The [README](../README.md) carries the headline numbers; this
document carries the reasoning, the caveats, and everything needed to reproduce
or dispute them.

Corpus background and the superseded v1 results live in
[CORPUS_HISTORY.md](CORPUS_HISTORY.md).

---

## Two Independent Harnesses

Quality is measured two ways on purpose, because each covers the other's blind
spot.

| | Graded relevance eval | Known-item eval |
|---|---|---|
| Script | `scripts/eval_redesign.py` | `python -m src.eval.known_item_eval` |
| Question | *Given an exploratory query, how good is the ranking?* | *Given an exact title, is that book #1?* |
| Labels | LLM judge, graded 0/1/2 | Objective — the sampled book either comes back first or it does not |
| Strength | Measures the actual product | No judge, no pooling, no subjectivity; verifiable by hand |
| Weakness | Depends on a judge that agrees with a human ~2/3 of the time | Easiest possible task; not representative of real queries |

A claim is only treated as solid here when both agree, or when the one that
applies is explicitly named.

---

## Methodology

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

## Retrieval Quality

**v2 corpus (84,801 books), n=98, k=10.** Two of 100 queries (one `author`, one
`title_lookup`) were dropped for having no relevant document in the pool.
Confidence intervals are 1,000-replicate percentile bootstraps.

| Mode | MRR@10 | NDCG@10 | Recall@10 | Median latency | p90 |
|------|--------|---------|-----------|----------------|-----|
| Keyword (TF-IDF) | 0.889 [0.832, 0.937] | 0.633 [0.582, 0.682] | 0.331 [0.283, 0.380] | 141 ms | 172 ms |
| Vector (nomic-256d) | 0.951 [0.910, 0.986] | 0.750 [0.710, 0.790] | 0.392 [0.345, 0.441] | 212 ms | 229 ms |
| Hybrid (RRF) | 0.953 [0.915, 0.985] | 0.754 [0.714, 0.793] | 0.414 [0.361, 0.471] | 216 ms | 233 ms |
| **Hybrid + Rerank** | **0.982 [0.954, 1.000]** | **0.835 [0.799, 0.865]** | **0.450 [0.398, 0.501]** | 2,158 ms | 2,645 ms |

Recall@10 is bounded above by **0.559** on average, because a pool can hold more
than ten relevant documents while only ten can be returned. Recall figures are
therefore not comparable to systems evaluated with complete judgments — nor to
the v1 numbers, whose ceiling was 0.794 on a sparser pool.

**Vector and hybrid are statistically indistinguishable on this corpus** (NDCG
0.750 vs 0.754). That corroborates the known-item finding below, where hybrid
lost its rank-1 lead to pure vector on v2, and is the strongest evidence in this
repo that TF-IDF's lack of length normalization is now the binding constraint on
the lexical arm.

---

## Threshold Sensitivity: What 0.982 Actually Means

The table above counts a document as relevant at **grade ≥ 1** ("partially
relevant" or better). On this pool that is a lenient bar: **23.27 of 50** pooled
documents clear it, so simply shuffling the pool scores **0.598**. Quoting 0.982
without that denominator would be the most misleading number in this repo.

Recomputing every mode under a **strict** threshold counting only grade-2 ("fully
relevant") documents — same 98 queries, same ranked lists, same judgments:

| Mode | MRR@10 (grade ≥ 1) | MRR@10 (grade = 2) | Recall@10 (grade = 2) |
|------|--------------------|--------------------|-----------------------|
| *Random shuffle* | *0.598* | *0.231* | — |
| Keyword (TF-IDF) | 0.889 | 0.747 [0.668, 0.822] | 0.518 |
| Vector (nomic-256d) | 0.951 | 0.838 [0.769, 0.901] | 0.659 |
| Hybrid (RRF) | 0.953 | 0.844 [0.783, 0.899] | 0.672 |
| **Hybrid + Rerank** | **0.982** | **0.958 [0.913, 0.990]** | **0.736** |

Two things survive the harder bar:

- **The result holds.** 0.958 against a 0.231 random baseline is a **4.1×** lift,
  a far more meaningful claim than 0.982 against 0.598.
- **Reranking looks better under scrutiny, not worse.** Its margin over hybrid
  *grows* from **+0.029** at the lenient threshold to **+0.114** at the strict
  one. The reranker is not merely pulling *something* relevant to rank 1; it
  disproportionately pulls the *best* document to rank 1 — precisely what a
  lenient MRR is blind to. This replicates the same pattern seen on v1, more
  strongly.

Note the lenient bar is *weaker* on v2 than it was on v1 (23.27 qualifying
documents per pool vs 11.95), so the strict column is the one to read.

The random baseline is Monte Carlo over actual per-query pool sizes and relevant
counts. The lenient column is fetched from the live API **independently of the
headline table**, by a separate script, and now reproduces it exactly on all four
modes — which doubles as a reproducibility check on the main harness. Raw ranked
lists are committed to `data/eval/v2/rankings.json` so these numbers stay
verifiable after the corpus changes.

Reproduce with `python scripts/threshold_sensitivity.py --offline`.

---

## Paired Comparisons

| Comparison | NDCG@10 delta | 95% CI | MRR@10 delta | 95% CI |
|------------|---------------|--------|--------------|--------|
| Hybrid − Keyword | **+0.121** | [+0.087, +0.158] | +0.064 | [+0.017, +0.107] |
| Hybrid+Rerank − Hybrid | **+0.081** | [+0.054, +0.110] | +0.029 | [+0.005, +0.062] |
| Hybrid+Rerank − Keyword | **+0.202** | [+0.159, +0.250] | +0.094 | [+0.048, +0.144] |

All six intervals exclude zero. Eleven unadjusted intervals were computed in
total, so roughly 0.6 would be expected to exclude zero by chance; the observed
effects are far larger than that.

---

## Where Reranking Helps

| Category | n | Hybrid NDCG | +Rerank | Delta | 95% CI |
|----------|---|-------------|---------|-------|--------|
| author | 13 | 0.664 | 0.779 | +0.115 | [−0.002, +0.255] |
| combined | 9 | 0.756 | 0.808 | +0.053 | [−0.002, +0.124] |
| exploratory | 27 | 0.779 | 0.838 | +0.059 | [+0.011, +0.106] |
| genre_topic | 32 | 0.763 | 0.867 | **+0.104** | [+0.062, +0.151] |
| title_lookup | 17 | 0.767 | 0.829 | +0.062 | [+0.013, +0.118] |

Reranking points the same direction in all five categories, but the intervals for
**author** (n=13) and **combined** (n=9) both just include zero — those two are
underpowered, not evidence of no effect. Only `genre_topic` clears n=30 and is
individually well-powered; the rest should be read as directional.

---

## Ceiling Analysis

A perfect reranker over the same 25-candidate pool would score **0.940** NDCG
against hybrid's **0.754** — headroom of **+0.186**. The cross-encoder captures
**+0.081**, or about **44%** of what is theoretically available at that depth.

The remaining gap is a retrieval problem, not a ranking one: it can only be
closed by getting better candidates into the pool.

---

## Known-Item Accuracy

Sample a book from the index, search its exact title, check whether that book
comes back first. No LLM judge, no pooling, no subjective grading — the answer is
either right or wrong, and anyone can verify it by hand in seconds.

**Measured on the v2 84,801-book index** (50 sampled titles):

| Mode | Acc@1 | Acc@5 | MRR | v1 Acc@1 (26.5K) |
|------|-------|-------|-----|------------------|
| **Vector (nomic-256d)** | **94%** | 96% | 0.950 | 100% |
| Hybrid (RRF) | 86% | **98%** | 0.917 | 94% |
| Keyword (TF-IDF) | 66% | 80% | 0.732 | 74% |

Every mode scores lower than on v1, the expected cost of an index **3.2×
larger**: the same query now competes against three times as many plausible
titles. The v1 column is shown for scale, not as a regression — the two numbers
describe different systems, which is why the v1 baseline was archived rather than
reused as a CI gate.

**The interesting result is that hybrid no longer wins at rank 1.** On v1, hybrid
(94%) sat between vector (100%) and keyword (74%). On v2 the keyword arm degrades
to 66%, and RRF propagates that: hybrid's Acc@1 (86%) falls *below* pure vector
(94%). Hybrid still has the best top-5 recovery (**98%** vs vector's 96%), so
fusion is still buying recall — at a cost in top-1 precision that did not exist at
26K. This is the concrete evidence behind "TF-IDF over BM25" being the closest
call in the project.

Keyword's failures are concentrated rather than random: **74%** accuracy on
distinctive titles (31/42) vs **25%** on titles made of common words (2/8).
Inherent to lexical scoring, not an indexing defect.

**30 hard variants** — typos, partial titles, title-plus-author forms:

| Mode | Acc@1 | Acc@5 |
|------|-------|-------|
| Hybrid (RRF) | **73%** | **87%** |
| Vector | **73%** | 80% |
| Keyword (TF-IDF) | 67% | 87% |

Degradation is graceful rather than catastrophic. This eval doubles as a CI
regression gate: every deploy re-runs all four modes against the live container
and fails the build if **hybrid or hybrid+rerank** accuracy drops more than 5
points below the recorded baseline.

### Reranking on v2: a paired test

`--modes hybrid,hybrid+rerank` runs both arms over identical queries against the
identical index in a single pass, so the reranker is the only variable.

| Fixture | Mode | Acc@1 | Acc@5 | MRR |
|---------|------|-------|-------|-----|
| Standard (n=50) | Hybrid | 86.0% | 98.0% | 0.913 |
| Standard (n=50) | **Hybrid + rerank** | **98.0%** | **100.0%** | **0.987** |
| Hard variants (n=30) | Hybrid | 73.3% | 86.7% | 0.800 |
| Hard variants (n=30) | **Hybrid + rerank** | **90.0%** | **96.7%** | **0.925** |

An average can hide a reranker that wins often and loses badly. The paired
breakdown shows it does not:

| Fixture | Fixed by rerank | Broken by rerank | McNemar exact *p* |
|---------|-----------------|------------------|-------------------|
| Standard | 6 | **0** | 0.031 |
| Hard variants | 5 | **0** | 0.0625 |

**Zero regressions across 80 paired queries.** The hard-variant *p* of 0.0625
looks like a failure to reach significance, but it is the *floor* for five
discordant pairs pointing the same way (2 × 2⁻⁵) — as significant as that sample
size permits, not weak evidence.

Most gains are **rank 2 → 1**, the fine-grained top-of-list discrimination a
cross-encoder is supposed to provide rather than a wholesale reshuffle. Two
documents were promoted from *outside* the top 10 — evidence the 2.5× candidate
pool does real work, since a reranker confined to the original top 10 could not
have found them.

---

## Reranker Performance Engineering

Reranking costs **~2.2 s end-to-end vs ~216 ms** for plain hybrid, which is why
it stays opt-in.

That cost used to be roughly twice as high. The cross-encoder scored all 25
candidates in one batch with `padding=True`, so a single long description set the
tensor width for every short one — **45.2% of processed tokens were padding**.
Sorting candidates by length and scoring them in sub-batches of 4, each padded
only to its own longest member, cuts padding to 17.7% (33.4% fewer tokens).
Because padding is masked by `attention_mask`, this changes no score: measured max
absolute difference vs. the flat batch is **exactly 0.0** across 12 trials of 25
real candidates, with zero ranking changes.

| | Flat batch | Length-bucketed | Change |
|---|---|---|---|
| Rerank stage (production) | 3642 ms | **1812 ms** | **2.01× faster** |
| End-to-end (production) | 4096 ms | **2089 ms** | 1.96× faster |
| Padding share of tokens | 45.2% | **17.7%** | 33.4% fewer tokens |

Batch size 4 came from a measured sweep. At 25 candidates the curve is flat
between 2 and 6 (676–781 ms locally); 4 captures nearly all of the win with half
the ONNX Runtime invocations of 2, which is the cost most likely to grow on a
smaller container. Setting it to 25 reproduces the flat-batch time exactly, the
sanity check that one bucket spanning everything *is* the old behaviour.

The riskiest part is the scatter-back: returning scores in sorted order would
mis-attribute every score while still producing a plausible-looking ranking.
`tests/test_reranker_batching.py` covers it with fakes rather than the 86 MB ONNX
model, so it runs on a clean clone — and reverting the scatter to sequential
assignment fails 8 of its 11 tests.

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

Stated plainly, because they bound what the tables above can support.

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
  calibrated judge in `src/eval/judge.py` — `scripts/eval_redesign.py` uses its
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
python scripts/eval_redesign.py --step all

# Individual steps
python scripts/eval_redesign.py --step generate   # corpus-grounded queries
python scripts/eval_redesign.py --step pool       # union candidates across modes
python scripts/eval_redesign.py --step judge      # LLM grading (costs tokens)
python scripts/eval_redesign.py --step eval       # metrics, no judging cost

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
| `data/eval/v2/*.v1-openlibrary.json` | The archived, superseded v1 run |
