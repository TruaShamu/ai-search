# Corpus

What the dataset is, how it was cleaned, and why it replaced the original.

---

## Current Corpus (v2)

**84,801 English-language books** from a single Goodreads source where title,
author, and description all come from the same record. This eliminates an entire
class of data error by construction (see [v1 data defect](#v1-data-defect)
below).

### Cleaning

| Step | What it fixes | Measured impact |
|---|---|---|
| **Mojibake repair** (`ftfy`, cp1252→UTF-8) | Curly quotes, accented characters rendered as `Ã©`, `â€™` | 26.16% of source rows → 0.000% post-repair |
| **Language filtering** | Non-English books pollute the embedding space — a French translation of an unrelated book outscores a genuine English near-match (0.571 vs 0.274 cosine) | 8.37% of rows dropped |
| **Deduplication** | Duplicate work IDs | 0 duplicates remain |

The correct mojibake inverse is **cp1252, not latin-1** — latin-1 silently
leaves Windows-1252 punctuation (curly quotes, em dashes) still broken, which is
most of the damage in book descriptions.

### Derived Artifacts

Every artifact below is regenerated together when the corpus changes — they
desync silently if left stale:

- **Dense embeddings** — 170 shards, 100% coverage, all norms 1.0000
  (see [EMBEDDING.md](EMBEDDING.md))
- **TF-IDF vectorizer** — globally fit over the full corpus
- **SymSpell dictionary** — 67,263 terms from titles, authors, and subjects
- **Eval fixtures** — queries, pooled candidates, and judgments versioned to
  the corpus they were built against (see
  [EVAL_METHODOLOGY.md](EVAL_METHODOLOGY.md))

---

## v1 Data Defect

The original 26,519-book corpus (OpenLibrary) had a systemic data-quality defect
that the eval could not detect. Finding it was the most useful outcome of the
human labeling round.

**The problem.** Only 5.4% of OpenLibrary works carry a description, so an
augmentation step joined Goodreads descriptions by normalized title. That
dataset has no author column, so on a title collision the longest description
won — failing **open** by giving confident-looking wrong answers.

**Scale.** 3,383 records (12.8% of the index) provably carried a description
belonging to a different author's book. *The Ugly Duckling* by Andersen was
described as a thriller about an assassination attempt. 12.8% is a floor — it
counts only detectable collisions where two books share one description.

**Why the eval missed it.** The eval is a closed loop: a wrong description is
embedded, retrieved for the query it matches, and judged relevant against *the
text shown*. Mean grades by provenance were statistically indistinguishable
(0.268 vs 0.263). Only a human comparing a description against its own title
could catch this.

**Why migration, not a patch.** Failing closed on collisions is insufficient —
some title matches are 1:1 yet still cross-author (Andersen's *Ugly Duckling*
matched the only Goodreads entry with that title). With no author or ISBN in
the source, the error is unfixable in principle. Migrating to a single-source
dataset makes this class of error **impossible to represent**.

---

## v1 → v2 Comparison

| Property | v1 | v2 |
|---|---|---|
| Records | 26,519 | **84,801** |
| Cross-author corruption | ≥12.8% (floor) | **0 by construction** |
| Mojibake | 26.16% | **0.000%** |
| Duplicate work IDs | present | **0** |
| Languages | mixed, unmeasured | English only |

> **Absolute numbers are not comparable across v1 and v2.** The v2 index is
> 3.2× larger, making known-item retrieval strictly harder. A lower v2 score is
> the expected cost of a corpus that is 3× bigger and no longer lying about
> which book a description belongs to.

Eval fixtures are versioned alongside the corpus. v1 gold IDs have 0% overlap
with the v2 ID space (`gr:` namespace) — reusing old judgments would silently
score ~0 on every query. Both generations are archived side by side so claims
remain verifiable. The human-labelled judge validation
(`data/eval/v2/*.v1-openlibrary.json`) is preserved because hand labels cannot
be regenerated and still back the κ figure in
[EVAL_METHODOLOGY.md](EVAL_METHODOLOGY.md#judge-validation-and-known-limitations).

### Superseded v1 Results

Preserved for auditability. **Do not quote as current** — see
[EVAL_RESULTS.md](EVAL_RESULTS.md) for v2 numbers.

Measured on 26,519 books, n=94 queries:

| Mode | MRR@10 | NDCG@10 | Recall@10 |
|------|--------|---------|-----------|
| Keyword (TF-IDF) | 0.851 | 0.593 | 0.440 |
| Vector (nomic-256d) | 0.896 | 0.698 | 0.514 |
| Hybrid (RRF) | 0.910 | 0.703 | 0.523 |
| Hybrid + Rerank | 0.985 | 0.801 | 0.583 |

> Relative comparisons between modes are unaffected by the data defect — every
> mode searched the same index. What the defect undermines is the absolute claim
> that a top-ranked result is the *right book*.

