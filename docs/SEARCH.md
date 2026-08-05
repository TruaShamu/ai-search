# Search Design

How BookSearch finds and ranks results: the two retrieval arms, how they are
fused, query understanding, and the limitations of each.

---

## Retrieval Arms

Every search starts with one or both of two independent retrieval arms, each
scoring documents from a different signal.

### Dense (Vector) Search

The query is embedded with nomic-embed-text-v1.5 (dim=256, Matryoshka) using the
`search_query:` prefix convention, then compared against pre-computed document
vectors by cosine similarity. This is what makes *"a heist that goes wrong"*
return *Freezer Burn* — the model learned that heists and crime fiction occupy
nearby regions in embedding space despite sharing no words.

**Strengths:** Semantic understanding, paraphrase invariance, handles queries
that share no vocabulary with the target document.

**Limitations:** Weak on exact-match lookups. A query for "1984" retrieves books
*about* dystopia rather than the specific book titled *1984* — the embedding
represents meaning, not identity. Measured: dense wins known-item Acc@1 (94%)
over hybrid (86%), but that is because the keyword arm's failures pull hybrid
*down*, not because dense is good at exact match.

### Sparse (TF-IDF) Search

The query is transformed by the same `TfidfVectorizer` that was fit over the
corpus during indexing (see [EMBEDDING.md](EMBEDDING.md#sparse-arm)), producing a
sparse vector of term weights. Documents are scored by dot product — terms that
are rare in the corpus get high weight, common terms get low weight.

**Strengths:** Exact-match precision. A search for an author name or a
distinctive title word retrieves exactly the documents containing that string.

**Limitations:**

- **No length normalization.** TF-IDF does not penalize long documents the way
  BM25 does. A book with a 2,000-word description containing a query term once
  scores similarly to a book with a 50-word description containing it once —
  BM25's saturation curve would downweight the former. At 84.8K documents this
  is now the closest design call in the system: the eval shows keyword Acc@1 at
  66% (vs BM25's expected ~70-75%), and this is the binding constraint on the
  lexical arm.
- **No semantic understanding.** "Heist" and "robbery" are unrelated terms. A
  query about one will not retrieve documents using the other.
- **Common-word titles are unsearchable.** Keyword accuracy on distinctive titles
  is 74% vs 25% on titles made of common words. Inherent to lexical scoring.

---

## Hybrid Fusion (RRF)

When `mode=hybrid`, both arms run independently and their results are fused
server-side in Qdrant using **Reciprocal Rank Fusion**. Each document receives:

```
score = Σ  1 / (k + rank_i)
```

where the sum is over each arm's ranked list the document appears in, and `k` is
a constant (Qdrant uses k=60 by default). A document ranked #1 in both lists
scores `2/(60+1) = 0.0328`; a document ranked #1 in one and absent from the
other scores `1/(60+1) = 0.0164`.

RRF has a useful property: it requires no score calibration between arms. TF-IDF
scores and cosine similarities live on different scales, but RRF uses only
*ranks*, so the arms are automatically comparable.

### Limitations of RRF

- **Fusion averages in your weakest arm.** If keyword confidently puts the wrong
  document at #1, that document gets a `1/(k+1)` boost regardless of how wrong
  it is — RRF has no mechanism to trust one arm over the other. This is why
  hybrid Acc@1 (86%) falls below pure vector (94%): the keyword arm's failures
  drag fusion down.
- **Tied scores are structurally inevitable.** Two documents at the same rank in
  exactly one input list receive bit-identical RRF scores. Dense cosine scores
  effectively never collide; RRF scores routinely do. This caused 8 of 40
  queries to return a different #1 book on back-to-back requests until a
  client-side tie-break was added (see
  [EVAL_METHODOLOGY.md](EVAL_METHODOLOGY.md#determinism)).
- **No learned weighting.** The contribution of each arm is fixed by `k`. A
  learned fusion model could weight arms per-query (trust keyword more for title
  lookups, dense more for exploratory queries), but RRF cannot.
- **Recall is the real payoff, not precision.** Hybrid's NDCG (0.754) barely
  exceeds vector's (0.750), but its Recall@10 (0.414 vs 0.392) and Acc@5 (98%
  vs 96%) are consistently higher. Fusion is buying coverage at the cost of
  top-1 precision.

---

## Query Understanding

Before retrieval, the raw query passes through a lightweight pipeline
(`src/query/pipeline.py`) that runs in <1ms:

**1. Spell correction** (`src/query/spell.py`). SymSpell with a corpus-derived
dictionary (67,263 terms from titles, authors, and subjects) plus the standard
English frequency dictionary. Uses the symmetric delete algorithm — O(1) lookups
after the dictionary is built. Max edit distance 2.

**2. Intent classification** (`src/query/intent.py`). Rule-based pattern matching
that classifies the query into one of five intents and routes to the optimal
search mode:

| Intent | Example | Routes to |
|---|---|---|
| `keyword` | ISBN, exact title in quotes | keyword mode |
| `semantic` | "books about loneliness" | vector mode |
| `author` | "books by Ursula Le Guin" | keyword mode (author field match) |
| `similar_to` | "books like Project Hail Mary" | vector mode (embed the reference) |
| `filtered` | "fantasy novels from the 1990s" | hybrid + year filter |

### Limitations

- **Rule-based, not learned.** Intent classification uses regex patterns, not a
  trained model. It catches common phrasing ("books like X", "by Author Name")
  but misses uncommon formulations or ambiguous queries. There is no confidence
  threshold — a non-matching query defaults to `semantic` intent with
  `mode=hybrid`.
- **No query expansion.** An earlier version had LLM-powered query expansion but
  it was removed after measuring **−21% NDCG** — the expanded terms introduced
  noise that hurt retrieval more than the broader recall helped.
- **Spell correction is dictionary-bounded.** SymSpell can only correct toward
  words in its dictionary. A misspelled author name not in the corpus dictionary
  will not be corrected. The corpus dictionary is regenerated when the index is
  rebuilt, so it stays in sync with the searchable content.
- **No per-query arm weighting.** The intent classifier picks a mode (keyword,
  vector, or hybrid) but cannot weight the arms within hybrid — a title-lookup
  query gets the same RRF k=60 balance as an exploratory query.
