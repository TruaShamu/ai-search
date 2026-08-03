# Evaluation Improvement Plan

> **Status: completed and superseded.** Every item in this plan was built. The
> heuristic annotator was replaced by an LLM judge, the judge was calibrated against
> 89 hand-labeled pairs (Cohen's kappa **0.314** — disclosed, not buried), the query
> set grew from 30 hand-written to **100 corpus-grounded** queries, and per-category
> breakdowns now ship with bootstrap confidence intervals.
>
> The plan also predates two later corrections: the sparse arm is **TF-IDF, not
> BM25** as written below, and the original query set turned out to be the dominant
> source of error — it had been generated without reference to the corpus, so it
> asked for books the index did not contain.
>
> **Current results and methodology: [EVALUATION.md](EVALUATION.md).** Kept as a
> record of what the eval looked like before it was rebuilt.

## Current State

We have a working eval framework (`src/eval/`) with:
- 30 queries across 6 categories
- Automated heuristic annotation (pools top-10 from all strategies, checks keyword overlap)
- MRR@10, NDCG@10, Recall@10 metrics
- 4-strategy comparison (BM25, vector, hybrid, hybrid+rerank)

**Problem:** The heuristic annotator is biased toward keyword matching, which inflates BM25 scores and penalizes vector search (which finds semantically relevant docs using *different* words). We need better relevance judgments.

---

## Improvement Plan (Priority Order)

### 1. LLM-as-Judge (High Impact, Low Cost)

Use a language model to score `(query, document)` relevance on a 0-2 scale. This captures *semantic* relevance that keyword heuristics miss.

**Recommended model for judging:**

| Model | Input $/1M | Output $/1M | Quality | Best For |
|---|---|---|---|---|
| gpt-4.1-nano | $0.10 | $0.40 | Good | Ultra-budget fallback |
| **gpt-5.4-nano** | $0.20 | $1.25 | Very good | ✅ Default judge + RAG |
| gpt-5.4-mini | $0.75 | $4.50 | Excellent | Gold-standard calibration |
| gpt-4.1 | $2.00 | $8.00 | Best | Calibration baseline only |

**Estimated cost for our dataset:**
- 30 queries × ~20 pooled docs each = ~600 judgments
- ~200 tokens per judgment (prompt + response)
- Total: ~120K tokens input + ~30K output
- **gpt-5.4-nano: ~$0.06 total** (basically free)
- **gpt-5.4-mini: ~$0.25 total** (still cheap, use for calibration)

**Recommended approach:**
1. Use **gpt-5.4-nano** for bulk annotation (600 judgments, ~$0.06)
2. Use **gpt-5.4-mini** on 50 random samples as calibration baseline
3. Measure LLM-human agreement (Cohen's kappa) on 20 manually judged queries

**Prompt template:**
```
You are a search relevance judge. Given a user's search query and a book result, 
rate the relevance on this scale:

0 = Not relevant (wrong topic, misleading match)
1 = Partially relevant (related topic but not what the user wants)
2 = Highly relevant (directly answers the user's search intent)

Query: "{query}"
Book title: "{title}"
Author: "{author}"
Description: "{description}"
Subjects: {subjects}

Relevance (0/1/2):
Reasoning (1 sentence):
```

**Implementation:** `src/eval/llm_judge.py`
- Batch all (query, doc) pairs
- Call Azure OpenAI API (gpt-5.4-nano)
- Parse structured output (relevance + reasoning)
- Save annotated judgments to `data/eval/queries_llm_judged.json`

### 2. Manual Gold-Standard Judgments (20 queries)

Even with LLM-as-judge, manual judgments are the ground truth.

**Process:**
1. Pick 20 queries spanning all categories
2. For each: pool top-10 from all 4 strategies (unique ~25-30 docs)
3. Judge each doc: 0/1/2 relevance
4. Time estimate: ~1 hour total

**Output:** `data/eval/queries_gold.json` — the definitive eval set

**Use for:**
- Calibrating the LLM judge (measure agreement)
- Final reported numbers
- CI regression detection

### 3. Category-Specific Analysis

Break down metrics by query category to understand where each strategy excels:

```
| Category     | BM25 NDCG | Vector NDCG | Hybrid NDCG | +Rerank NDCG |
|--------------|-----------|-------------|-------------|--------------|
| genre        | 0.xx      | 0.xx        | 0.xx        | 0.xx         |
| topic        | ...       | ...         | ...         | ...          |
| concept      | ...       | ...         | ...         | ...          |
| specific     | ...       | ...         | ...         | ...          |
| era          | ...       | ...         | ...         | ...          |
| cross-domain | ...       | ...         | ...         | ...          |
```

**Expected findings:**
- BM25 wins on `specific` (author names, exact titles)
- Vector wins on `concept` (abstract themes like "loneliness")
- Hybrid wins on `cross-domain` and `genre` (needs both signals)
- Reranker helps most on `concept` (nuance requires cross-attention)

### 4. Statistical Significance

With 30 queries, we can do:
- Paired t-test or Wilcoxon signed-rank test between strategies
- Report p-values alongside metric differences
- "Hybrid+rerank improves NDCG@10 by 0.08 over hybrid alone (p<0.05)"

### 5. Expand Query Set (Phase 3)

Target 100+ queries for publication-grade eval:
- 30 current queries (annotated)
- +30 from OpenLibrary reading lists (scrape curated lists → natural queries)
- +20 synthetic from GPT (generate queries that a user might type)
- +20 adversarial (misspellings, ambiguous, multi-intent)

### 6. CI Integration

```yaml
# .github/workflows/eval.yml
on:
  pull_request:
    paths: ['src/search/**', 'src/reranker/**', 'src/azure_search/**']

jobs:
  eval:
    steps:
      - run: python -m src.eval.run --no-rerank
      - compare: results.json vs baseline (fail if NDCG drops >5%)
      - comment: post eval table to PR
```

---

## Priority Execution Order

| Step | Effort | Impact | Do When |
|---|---|---|---|
| LLM-as-judge annotation | 1 hour | ⭐⭐⭐ | Now |
| Manual gold-standard (20 queries) | 1 hour | ⭐⭐⭐ | This week |
| Category breakdown | 30 min | ⭐⭐ | After annotation |
| Statistical significance | 15 min | ⭐ | After gold judgments |
| Expand to 100 queries | 2 hours | ⭐⭐ | Phase 3 |
| CI integration | 1 hour | ⭐⭐ | Phase 3 |

---

## Model Recommendations Summary

For **LLM-as-judge**: Use **gpt-5.4-nano** (~$0.06 for entire eval set). Cheap enough to re-run on every PR, better reasoning than older nano models.

For **RAG /ask endpoint**: Use **gpt-5.4-nano** ($0.20/1M input, $1.25/1M output) — best quality-to-cost ratio in the nano class, 1M+ context window.

For **query expansion/understanding**: **gpt-5.4-nano** — consistent model across all LLM features simplifies deployment.

**Budget impact:** All LLM usage combined < $1/month at our query volume. Negligible.
