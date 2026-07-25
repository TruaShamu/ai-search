# Project Progress Log

## Project: AI-Powered Book Search (OpenLibrary)
**Goal:** Portfolio piece for backend + infra AI engineering roles
**Stack:** Python, PyTorch, FastAPI, Azure (AI Search, Container Apps, Foundry)

---

## 2026-07-19 — Project Kickoff + Phase 0

### Architecture & Design
- [x] Created `DESIGN.md` — full system architecture with 10 components
- [x] Architecture: ETL → Enrichment → Embedding → AI Search (hybrid+RRF) → Reranker → RAG
- [x] Budget analysis: fits in $150/mo Azure VS Enterprise credits (~$115-125/mo)
- [x] Phased development plan (Phase 0-5)

### Key Design Decisions
| Decision | Choice | Rationale |
|---|---|---|
| Vector store | Azure AI Search | Built-in hybrid BM25+vector+RRF; engineer around it |
| Embedding model | nomic-embed-text-v1.5 | Best speed/quality/size ratio (see evaluation doc) |
| Embedding strategy | Tiered templates + Nomic task prefixes | Maximizes signal per data quality tier |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Runs on CPU, scores 50 candidates in <500ms |
| RAG model | gpt-4.1-nano | 3x cheaper than gpt-4o-mini, good enough for grounded answers |
| Cache | Redis sidecar (Container Apps) | $0 extra, scale-to-zero |
| IaC | Bicep | Native Azure, aligns with the Azure-focused portfolio |

### EDA Findings
- [x] Dataset: `storytracer/openlibrary_dump_2024-04-30` on HuggingFace (Parquet)
- [x] Total works in dump: **34,666,230**
- [x] Only **5.5% have descriptions** — biggest data quality challenge
- [x] 65.8% have subjects, 92.8% have author refs, 17.6% have year
- [x] Descriptions are JSON-encoded (`{"type":"/type/text","value":"..."}`)
- [x] Subject tags need normalization (case variants: "Fiction" vs "FICTION")
- [x] Authors stored as keys — need join with authors dump (12.7M authors)
- [x] Description quality: 71.6% good narrative, 18.8% weak, 4.9% junk
- [x] Tiered data strategy:
  - Tier 1 (title+desc+subjects): ~1.6M works
  - Tier 2 (title+subjects): ~21M works
  - Tier 3 (title only): ~11M works

### ETL Pipeline
- [x] Built `src/etl/pipeline.py` — full ETL orchestration
- [x] Built `src/etl/clean.py` — description parsing, subject normalization, year extraction
- [x] Built `src/etl/schema.py` — Pydantic `Book` model with `embedding_text()` method
- [x] Author name resolution: 12.7M author key→name lookup (loaded in ~40s)
- [x] English language filter for descriptions
- [x] Junk description filter (page counts, too-short)
- [x] Exported `books_tier1-2_500k.jsonl` — **250,811 books** (87.2 MB)
  - 13,431 Tier 1 (with descriptions)
  - 237,380 Tier 2 (subjects only)
  - 95.4% have resolved author names

### Documentation
- [x] `DESIGN.md` — system architecture (10 components, infra, budget, phases)
- [x] `docs/embedding-model-selection.md` — 6-model comparison with benchmarks
- [x] `docs/embedding-strategy.md` — what/how to embed, field mapping
- [x] `docs/progress.md` — this file

### Project Structure
```
ai-search/
  DESIGN.md                           # System architecture
  docs/
    embedding-model-selection.md      # Model comparison (6 models)
    embedding-strategy.md             # What/how to embed
    progress.md                       # This file
  src/
    etl/
      __init__.py
      schema.py                       # Book pydantic model
      clean.py                        # Text cleaning utilities
      pipeline.py                     # ETL orchestration
  data/
    processed/
      books_tier1_200k.jsonl          # 5,923 Tier 1 books (test)
      books_tier1-2_500k.jsonl        # 250,811 books (main dataset)
  notebooks/
    01_eda.py                         # Initial EDA
    01_eda_part2.py                   # Subjects, dates, filtering
    01_desc_quality.py                # Description quality analysis
```

---

## Phase 0 — FAISS Prototype ✅ COMPLETE
- [x] Installed nomic-embed-text-v1.5 (sentence-transformers)
- [x] Batch embedded Tier 1 books (13,431 with descriptions) — 38 min on CPU
- [x] Built FAISS index (384d Matryoshka, 19.7 MB)
- [x] Wired up FastAPI `/search`, `/health`, `/stats` endpoints
- [x] Tested with 8 sample queries — results validated as good quality
- [x] Served FastAPI, confirmed endpoints work end-to-end via curl

---

## Phase 1 — Azure AI Search (Hybrid) — IN PROGRESS

### 2026-07-19 — Azure AI Search Setup
- [x] Provisioned free-tier AI Search: `booksearch-ais` (East US)
- [x] Resource group: `ai-search-rg`
- [x] Created index `books-v1` with hybrid config (BM25 + HNSW vector + RRF)
- [x] Pushed 13,431 docs + 384d vectors (51 MB, fits free tier)
- [x] Built `src/azure_search/` module (index.py, push.py, search.py)
- [x] **Hybrid search working!** Tested BM25-only, vector-only, and hybrid (RRF)

### Hybrid Search Quality Comparison
| Query | BM25 Top-1 | Vector Top-1 | Hybrid Top-1 |
|---|---|---|---|
| "romance set in Scotland" | Desmond goes to Scotland ❌ | Seducing the Highlander ✅ | Scottish Magic ✅ |
| "history of computing" | Data management and Internet... ⚠️ | The Internet in Everyday Life ⚠️ | The Internet in Everyday Life ⚠️ |
| "world war 2 memoir" | At battle in World War II ✅ | War-time diary of Gertrude Bathurst ✅ | At battle in World War II ✅ |

**Key finding:** RRF hybrid consistently promotes results that have BOTH keyword and semantic relevance, avoiding pure-keyword false positives.

### Storage Note
- 13K docs + 384d vectors = 51 MB (just over free tier's 50 MB soft limit)
- Working fine now; may need Basic tier ($75/mo) for 250K docs

---

## Next Up (Priority Order)
1. [x] **Swap FastAPI backend → Azure AI Search** ✅ Done!
2. [x] **`/search/compare` endpoint** ✅ Done! (side-by-side BM25 vs vector vs hybrid)
3. [x] **Cross-encoder reranker** ✅ Done! (ms-marco-MiniLM-L-6-v2, ~330ms on CPU)
4. [x] **ONNX export of reranker** ✅ Done! (1.6x speedup, 166ms for 50 candidates)
5. [ ] ONNX export of Nomic embedding model (for ACA ingestion worker)
6. [ ] Full Tier 1+2 embedding on ACA (250K books)

### Reranker Results
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params)
- PyTorch: ~266ms for 50 candidates on CPU
- **ONNX: ~166ms for 50 candidates (1.6x speedup)**
- Wired into `/search?rerank=true` — retrieves 5x candidates, reranks, returns top_k
- Key win: promotes semantically relevant results that keyword matching misses
  - "romance in Scotland": Seducing the Highlander jumped from #7 → #1
  - "cooking italian food": Mediterranean cuisine jumped from #4 → #1
  - "loneliness and isolation": "Single is not a curse" jumped from #18 → #5
- Full pipeline (warm): ~560ms total (280ms retrieval + 280ms rerank)

### ONNX Export
- Reranker: exported via `optimum` (ORTModelForSequenceClassification)
- Location: `data/models/reranker-onnx/` (87 MB)
- Nomic embedding: blocked — custom NomicBert architecture has rotary embedding ops that fail standard export. Will use sentence-transformers ONNX backend or skip for now.

---

## Evaluation Framework ✅ Built

### Structure
- `src/eval/dataset.py` — 30 eval queries across 6 categories (genre, topic, concept, specific, era, cross-domain)
- `src/eval/metrics.py` — MRR@k, NDCG@k, Recall@k, Precision@k
- `src/eval/annotate.py` — pooling-based auto-annotation (pools top-10 from all modes, heuristic relevance)
- `src/eval/run.py` — full harness comparing all 4 strategies

### Initial Results (heuristic judgments — biased toward keyword matching)
```
Strategy             MRR@10     NDCG@10    Recall@10    Latency
keyword              1.0000     0.8583     0.7132       160ms
vector               0.9111     0.5749     0.4254       164ms
hybrid               0.9344     0.7157     0.5901       226ms
hybrid+rerank        0.8815     0.5980     0.4347       1199ms
```

⚠️ **These results were misleading** — the heuristic annotator checked literal keyword overlap, inflating BM25 and penalizing vector search.

### LLM-as-Judge Results ✅ (gpt-5.4-nano, 541 judgments, $0.058)

```
Strategy             MRR@10     NDCG@10    Recall@10    Latency
keyword              0.8667     0.6031     0.5144       157ms
vector               0.9667     0.7943     0.6759       164ms
hybrid (RRF)         0.9444     0.8004     0.6591       234ms
hybrid+rerank        0.9011     0.6304     0.4715       918ms
```

### Key Findings (with proper semantic judgments)
1. **Vector search dominates BM25** — NDCG +32%, Recall +31%. Semantic matching wins.
2. **Hybrid (RRF) is best overall** — NDCG 0.800, combining both signals optimally.
3. **Reranker hurts performance** — ms-marco cross-encoder was trained on web passages, not short book metadata. It incorrectly demotes relevant books that have terse descriptions.
4. **Heuristic vs LLM agreement was only 35%** — proves keyword-overlap is not a valid relevance proxy for semantic search evaluation.

### Reranker Analysis — Deep Dive

Tested two rerankers with **full descriptions** (all 13K books are Tier 1 with 800+ char descriptions):

```
Strategy                    MRR@10    NDCG@10    Recall@10    Latency
hybrid (no rerank)          0.9444    0.8004     0.6591       234ms
hybrid + ms-marco (ONNX)    0.8944    0.6961     0.5649       698ms     ← -13% NDCG
hybrid + BGE-v2-m3          0.8611    0.6695     0.5257       23,449ms  ← -16% NDCG
```

**Root cause:** Both rerankers hurt because hybrid+RRF already has high precision at top-10 for a 13K corpus. Rerankers add value when first-stage retrieval has many false positives; here, the fusion of BM25 + vector already gives clean top-10 results. The cross-encoders shuffle good results down.

**When rerankers will help:**
- At 250K+ books (more noise in retrieval pool → reranker adds signal)
- With fine-tuned model (trained on book metadata, not web passages)
- The reranker code and ONNX pipeline are preserved for this future use

### Category Breakdown (NDCG@10, LLM-judged)

```
Category        N   BM25    Vector  Hybrid  Winner
concept         5   0.480   0.792   0.796   Hybrid
cross-domain    5   0.501   0.765   0.737   Vector
era             2   0.234   0.689   0.773   Hybrid
genre           5   0.586   0.826   0.879   Hybrid
specific        4   0.797   0.754   0.756   BM25
topic           9   0.733   0.835   0.820   Vector
```

**Insights:**
- BM25 wins only on `specific` queries (exact author/title matching)
- Vector wins on `topic` and `cross-domain` (semantic understanding)
- Hybrid wins on `concept`, `genre`, `era` (needs both signals combined)
- BM25 collapses on abstract queries: `concept` (0.48), `era` (0.23)

### TODO
- [x] LLM-as-judge for automated semantic relevance scoring ✅
- [x] Category-breakdown analysis ✅
- [x] Reranker deep-dive (two models, full descriptions) ✅
- [ ] Manual relevance judgments on 20 queries (gold standard calibration)
- [ ] Add eval to CI pipeline (regression detection)

---

## Query Understanding — Evaluated ✅

### What We Built
- SymSpell spell correction (English + 32K domain terms from index)
- Rule-based intent classifier (keyword/semantic/author/similar_to/filtered)
- Intent-based adaptive routing (mode selection per intent)
- LLM query expansion (gpt-5.4-nano, 3-5 synonym terms)

### What Actually Works

**Spell correction: ✅ SHIPS (only QU component with proven value)**
```
Condition                      NDCG@10    vs Clean
Clean (no typos)               0.800      baseline
Typos (no correction)          0.502      -37.2%
Typos + spell correction       0.757      -5.5%
```
- Recovers **85% of NDCG lost** to typos
- 80% exact query recovery rate (24/30)
- <1ms latency, pure CPU (SymSpell dictionary lookup)

**Intent routing: ❌ DROPPED (NDCG-neutral)**
```
Baseline (hybrid for all)      0.800
Routed (intent-adaptive)       0.800      +0.0%
```
- 93% of eval queries classify as `semantic` → route to hybrid anyway
- Year filters from NLU hurt when ground-truth doesn't match constraint
- Routing adds value only for ISBN (rare) and "books like X" (needs diverse eval set)

**LLM query expansion: ❌ NOT SHIPPING (-21% NDCG)**
```
Baseline (no expansion)        0.800
LLM expanded                   0.632      -21.0%
```
- Regresses **25/30 queries**
- Root cause: expansion terms dilute BM25 signal in hybrid RRF
- Vector component already captures synonyms via embeddings — expansion double-counts semantic signal
- +993ms latency (too slow regardless of quality)
- Only helps 4 queries where BM25 vocabulary mismatch was severe

### Current State (shipped)
- Spell correction → active (corrects query before search)
- Intent classification → metadata only (returned in response JSON for observability/analytics)
- No routing behavior, no query expansion, no hard filters from NLU

### Lessons Learned
1. Hybrid RRF at small scale is remarkably robust — additional components (reranker, expansion, routing) all failed to improve it
2. The vector component already handles synonyms/semantics — don't duplicate that signal on BM25 side
3. Rule-based intent classification without query logs or user signals is guesswork
4. Spell correction is the only QU component that addresses a real failure mode (typos degrade embedding quality)

### Future Work (if pursuing intent as a data project)
- Collect query logs with click/engagement signals
- Fine-tune distilbert on domain-specific intents
- Key papers: Guo & Lan 2020 (survey), Liu et al. 2020 (LinkedIn deep intent), Broder 2002 (taxonomy)

---

## Backlog (Phase 2-3)
- [x] Evaluation framework (MRR, NDCG@10) ✅
- [x] RAG `/ask` endpoint (gpt-5.4-nano) ✅
- [x] Query understanding evaluation ✅
- [ ] RAG eval (faithfulness, groundedness, relevance, completeness — LLM-as-judge)
- [ ] Caching layer (in-process LRU → Redis sidecar)
- [ ] Dockerfile + container setup
- [ ] Infrastructure-as-Code (Bicep)
- [ ] CI/CD (GitHub Actions)
- [ ] Monitoring + dashboards

## Stretch (Phase 5)

### Advanced Retrieval
- [ ] SPLADE learned sparse retrieval (replace BM25 with PyTorch model)
- [ ] ColBERT late-interaction retrieval
- [ ] Distillation: train fast bi-encoder from cross-encoder teacher
- [ ] Fine-tune embeddings on book pairs (contrastive learning on click data)
- [ ] TurboQuant + Matryoshka stacking (21x compression, fit 250K in free tier)

### Search Features (Exploration Ideas)
- [ ] **"More like this"** — `GET /similar?work_id=X` → nearest neighbors by embedding. Dead simple, very demoable.
- [ ] **Result explanations** — "Why did this match?" Show which signals contributed (BM25 score, vector similarity, matched subjects). Portfolio differentiator.
- [ ] **Faceted browse** — filter/aggregate by genre, decade, author. Exposes AI Search facets.
- [ ] **Autocomplete/typeahead** — prefix search on titles + authors. Good UX touch.
- [ ] **Personalization** — if we had user signals, weight results by reading history (overlaps with two-tower rec model).

### Data Quality
- [ ] LLM-generated descriptions for Tier 2/3 books
- [ ] A/B testing framework
