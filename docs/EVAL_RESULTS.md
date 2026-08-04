# Evaluation Results

Full result tables, analysis, and key findings from BookSearch's retrieval
evaluation. For how these numbers were produced and what they can and cannot
support, see [EVAL_METHODOLOGY.md](EVAL_METHODOLOGY.md).

---

## Key Findings

- **Hybrid search is not strictly better than dense retrieval — fusion averages in
  your weakest arm.** Everyone reports hybrid > vector. Here it isn't: pure vector
  wins known-item Acc@1 **94% vs 86%**, and on graded relevance the two are a
  rounding error apart (0.750 vs 0.754 NDCG). RRF ranks by *position*, so keyword's
  confidently-wrong #1 is worth 1/(k+1) no matter how wrong it is — fusion has no way
  to know which arm to trust. What hybrid actually buys is **recall** (0.414 vs 0.392,
  and 98% vs 96% Acc@5). It trades top-1 precision for coverage, and that trade got
  worse as the index grew 3.2×.
- **A relevance score is meaningless without the threshold it was computed at.** The
  same rankings score **0.982** counting grade ≥ 1 as relevant and **0.958** counting
  only grade 2 — but the random baseline moves **0.598 → 0.231**. So the honest claim
  goes from a 1.6× lift to a **4.1×** lift by tightening a definition, not by changing
  a line of code. Reranking's margin over hybrid grows nearly **4×** the same way
  (**+0.029 → +0.114**): a lenient bar makes a coarse filter look almost as good as a
  fine one, which systematically understates reranking.
  [Details](#threshold-sensitivity-what-0982-actually-means)
- **Rank fusion is structurally exposed to tie instability in a way dense retrieval is
  not.** RRF scores are sums of `1/(k+rank)`, so two documents at the same rank in
  exactly one input list collide on **bit-identical** scores. Ties are near-impossible
  among float cosine scores and routine under fusion — which is why **8 of 40 queries
  returned a different #1 book** on back-to-back identical requests while vector and
  keyword were perfectly stable. Anyone fusing rankings has this bug and probably has
  not looked. [Details](EVAL_METHODOLOGY.md#determinism)
- **Most of the remaining headroom is retrieval, not ranking.** A *perfect* reranker
  over the same 25 candidates would score 0.940 against hybrid's 0.754. The
  cross-encoder captures **44%** of that; the other 56% is unreachable by any
  reordering, because the right documents were never retrieved. Past this point a
  better reranker is the wrong investment. [Details](#ceiling-analysis)
- **An eval will happily measure its own query set instead of your system.** An
  earlier version put hybrid within noise of keyword. The queries had been
  LLM-generated with no view of the corpus, asking for *1984* and *The Great Gatsby*
  against an index of mostly obscure books — so many queries carried no gold document
  at all, and the modes became statistically indistinguishable (+0.003 MRR against a
  standard error near 0.07). Rebuilt from the corpus, the margin is a clear +0.121
  NDCG. [Details](EVAL_METHODOLOGY.md#how-queries-are-built)

Three of these surfaced as **bugs in my own code**: a `[:300]` passage truncation
that was discarding 60.2% of description text and making reranking look actively
harmful, an ingress timeout it was masking, and the tie instability above. The
harness has mostly paid for itself by contradicting me.

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
describe different systems, which is why the v1 baseline was never reused as a CI
gate. Its raw artifacts have since been deleted, so that column is a frozen
historical figure; see [Corpus History](CORPUS_HISTORY.md#the-superseded-v1-results).

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
