# Corpus History: v1 → v2

Background on the dataset behind BookSearch: where v1 came from, the data defect
that made it unusable for absolute claims, and what the v2 migration changed.

This is kept separate from the [main README](../README.md) and the
[evaluation methodology](EVALUATION.md) because it is *history* — it explains
why the current numbers look the way they do, and preserves the superseded v1
record so old claims stay auditable rather than quietly disappearing.

---

## Corpus Provenance, and a Data Bug the Eval Could Not See

v1 of the corpus had a data-quality defect worth stating in full. Finding it was
the most useful thing the human labeling round did.

**Where it came from.** All 26,519 books were OpenLibrary records, but only
**5.4%** of OpenLibrary works (13,431 of 250,811 scanned) carry a description —
and description is the highest-signal field for semantic retrieval. To raise
coverage, an augmentation step joined an external Goodreads description dataset
onto the corpus. That dataset (`booksouls/goodreads-book-descriptions`, 1.02M
rows) ships **only `title` and `description`** — there is no author column — so
normalized title was the only join key available. On a title collision,
`choose_better_description` kept the *longest* candidate. That fails **open**: an
ambiguous match yields a confident-looking wrong answer instead of no answer.

**What it cost.** Measured directly from the index:

| Description source | Records | Duplicate-description rate |
|---|---|---|
| OpenLibrary (native, joined on work ID) | 13,431 | **0.5%** |
| Goodreads (augmented, joined on title) | 13,088 | **26.5%** |

**3,383 records — 12.8% of the index — provably carried a description belonging
to a different author's book.** *The Ugly Duckling* by Hans Christian Andersen
was described as a thriller about an assassination attempt on a financier's
wife. **12.8% is a floor, not an estimate:** it counts only collisions where two
books share one description, which is the sole signature detectable from the
data. A further 9,743 Goodreads-sourced records matched a unique title and
cannot be verified either way.

**Why the eval scored it as fine.** The eval is a closed loop. A wrong
description is embedded, retrieved for the query it lexically matches, and then
read by the LLM judge — which grades it relevant, because *against the text it
was shown*, it is. Mean grades by provenance were statistically indistinguishable
(**0.268** OpenLibrary vs **0.263** Goodreads). More LLM judging would never have
surfaced this; only a human comparing a description against its own title could,
which is exactly how it was found. **583 of 5,000 judged pairs (11.7%) used a
provably corrupted description.**

**Why the fix was a migration and not a patch.** Failing closed on collisions is
the obvious repair, and it is insufficient: Goodreads holds only one *Ugly
Duckling*, so Andersen's title matched 1:1 with no collision to detect. This is a
**cross-corpus** mismatch, and with no author or ISBN in the source it is
unfixable in principle. The corpus was therefore migrated to a single-source
dataset carrying title, author, and description in the same record, which makes
this class of error **impossible to represent** rather than merely less likely.

---

## What the v2 Migration Changed

The index now holds **84,801 books** whose title, author, and description all
come from the same source row. Two further defects surfaced while building it,
both found by measuring rather than assuming:

| Property | v1 | v2 |
|---|---|---|
| Records | 26,519 | **84,801** |
| Cross-author description corruption | ≥12.8% (floor) | **0 by construction** |
| Mojibake (`Ã©`, `â€™`) | 26.16% of rows | **0.000%** |
| Duplicate work IDs | present | **0** |
| Languages | mixed, unmeasured | English only (8.37% dropped) |

**Mojibake.** 26.16% of source rows had been written as UTF-8 and re-read as
cp1252. The correct inverse is **cp1252, not latin-1** — latin-1 silently leaves
the Windows-1252 punctuation range (curly quotes, em dashes) still broken, which
is most of the damage in book descriptions. `ftfy` repairs it; the post-repair
rate is 0.000%.

**Language.** Embedding several languages into one space sounds harmless and is
not, for this corpus. Measured against the same English sentence: an English
paraphrase scores **0.787**, Spanish **0.635**, French **0.571**, German
**0.536**, and an unrelated English sentence **0.274**. A foreign-language
translation of an unrelated book therefore outranks a genuine English near-match,
and the cross-encoder reranker (English-only `ms-marco-MiniLM`) cannot repair it.
The 8.37% non-English rows are dropped.

**What was rebuilt.** Every corpus-derived artifact desyncs silently if left
stale, so all were regenerated together: dense vectors (170 shards, 100%
coverage, all norms 1.0000), the **globally-fit** TF-IDF vectorizer, the SymSpell
dictionary (45,606 → 67,263 terms), and the known-item fixtures. The v1
known-item baseline is archived rather than reused — it was measured on a 26.5K
index, so gating an 84.8K index against it would compare two different systems.

> **Absolute numbers are not comparable across v1 and v2.** The v2 index is
> **3.2× larger**, which makes known-item retrieval strictly harder: there are
> simply more plausible confusable titles. A lower v2 score is the expected cost
> of a corpus that is 3× bigger and no longer lying about which book a
> description belongs to.

---

## Eval Fixtures Are Corpus-Coupled

Eval artifacts are versioned alongside the corpus they were built against.
Unsuffixed filenames in `data/eval/v2/` are the current corpus; a
`.v1-openlibrary` suffix marks the superseded OpenLibrary run.

This is not tidiness. v1 gold document ids have **0% overlap** with the v2 id
space (`gr:` namespace), so the old judgments are structurally unrunnable against
the new index rather than merely stale. Re-using them would not error — every
query would simply match nothing and score ~0, silently halving every average.
The two generations are archived side by side so both remain verifiable.

---

## The Superseded v1 Results

Preserved so that claims made while v1 was live remain checkable. **Do not quote
these as current numbers** — see [EVALUATION.md](EVALUATION.md) for the v2
results.

Measured on the 26,519-book OpenLibrary index, n=94 queries (six `author`
queries dropped for having no relevant document pooled).

| Mode | MRR@10 | NDCG@10 | Recall@10 |
|------|--------|---------|-----------|
| Keyword (TF-IDF) | 0.851 [0.784, 0.911] | 0.593 [0.534, 0.649] | 0.440 [0.370, 0.508] |
| Vector (nomic-256d) | 0.896 [0.841, 0.946] | 0.698 [0.648, 0.743] | 0.514 [0.451, 0.569] |
| Hybrid (RRF) | 0.910 [0.858, 0.954] | 0.703 [0.660, 0.749] | 0.523 [0.459, 0.590] |
| Hybrid + Rerank | 0.985 [0.962, 1.000] | 0.801 [0.769, 0.835] | 0.583 [0.525, 0.641] |

Paired deltas: Hybrid − Keyword **+0.111 NDCG** [+0.079, +0.146]; Hybrid+Rerank −
Hybrid **+0.097** [+0.060, +0.135]; Hybrid+Rerank − Keyword **+0.208** [+0.159,
+0.260].

Strict grade-2 threshold: random baseline 0.111, keyword 0.659, vector 0.782,
hybrid 0.776, hybrid+rerank **0.881**. Mean achievable Recall@10 ceiling 0.794.

Per-category rerank NDCG delta: author +0.136 (n=12), combined −0.000 (n=10),
exploratory +0.113 (n=21), genre_topic +0.092 (n=32), title_lookup +0.116 (n=19).

Ceiling: hybrid 0.703 → oracle 0.917 at depth 25 (headroom +0.214); the
cross-encoder captured ~45%.

Known-item Acc@1 on v1: vector 100%, hybrid 94%, keyword 74%.

Raw v1 artifacts: `data/eval/v2/*.v1-openlibrary.json` and
`data/eval/v2/rankings_v1.json`.

> These are honest measurements of the system as deployed when taken, and the
> *relative* comparisons between modes are unaffected by the description defect —
> every mode searched the identical index, so a shared data defect cannot favor
> one over another. What the defect undermines is the absolute claim that a
> top-ranked result is the *right book*.

---

## Open Infrastructure Issue

Qdrant reports the collection as `status: red` with `IO Error: Input/output error
(os error 5)` from its segment optimizer. The storage volume is an Azure Files
(SMB) share, which does not give Qdrant the mmap and fsync semantics it expects;
this predates the migration (the v1 collection reported the same error).

Measured impact on serving: none detectable — all 84,801 points are present and
queryable, and warm hybrid latency is 182 ms median / 434 ms p95. A failed
optimization degrades toward exact search, which costs latency rather than
correctness. The real fix is block storage rather than SMB, which means a
workload profile the Consumption tier does not offer.
