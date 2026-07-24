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

### Key Insight
The heuristic annotator is biased toward BM25 because it checks literal term overlap. Manual inspection shows:
- Vector search consistently ranks **highly relevant (rel=2)** docs at the top
- BM25 gets more partial matches (rel=1) including false positives (e.g. "Computational logic and set theory" matching "romance **set** in Scotland")
- Proper manual judgments would likely show hybrid+rerank winning on NDCG

### TODO
- [ ] Manual relevance judgments on 20 queries (the gold standard)
- [ ] LLM-as-judge for automated but semantic relevance scoring
- [ ] Add eval to CI pipeline (regression detection)

---

## Backlog (Phase 2-3)
- [x] Evaluation framework (MRR, NDCG@10) ✅
- [ ] Query understanding (spell correction, intent classification)
- [ ] Caching layer (in-process LRU → Redis sidecar)
- [ ] RAG `/ask` endpoint (gpt-4.1-nano)
- [ ] Infrastructure-as-Code (Bicep)
- [ ] CI/CD (GitHub Actions)
- [ ] Monitoring + dashboards

## Stretch (Phase 5)
- [ ] SPLADE learned sparse retrieval (replace BM25 with PyTorch model)
- [ ] ColBERT late-interaction retrieval
- [ ] Distillation: train fast bi-encoder from cross-encoder teacher
- [ ] Fine-tune embeddings on click data
- [ ] LLM-generated descriptions for Tier 2/3 books
- [ ] A/B testing framework
