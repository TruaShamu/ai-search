# AI-Powered Book Search — Design Document

## Overview

Hybrid semantic search engine over the OpenLibrary catalog, demonstrating
backend + ML infra engineering on Azure. Combines BM25 keyword search with
vector similarity (via Azure AI Search), a PyTorch cross-encoder reranker,
query understanding, caching, and a RAG-powered answer endpoint.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA LAYER                             │
│                                                             │
│  OpenLibrary Dumps ──► ETL Pipeline ──► Document Enrichment │
│  (works, editions)     (Python)         (entity extraction, │
│                                          subject norm,      │
│                                          description gen)   │
│                                              │              │
│                                              ▼              │
│                                        Cleaned Corpus       │
│                                        (Azure Blob)         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   INDEXING PIPELINE                          │
│                                                             │
│  Cleaned Corpus ──► Batch Embedding ──► Push to Index       │
│                     (sentence-transformers,                  │
│                      CPU VM, all-MiniLM-L6-v2)              │
│                           │                                 │
│                           ▼                                 │
│                  Azure AI Search Index                       │
│                  ┌──────────────────┐                       │
│                  │ BM25 fields:     │                       │
│                  │   title, author, │                       │
│                  │   subjects, desc │                       │
│                  │   enriched_desc  │                       │
│                  │                  │                       │
│                  │ Vector field:    │                       │
│                  │   embedding[384] │                       │
│                  │                  │                       │
│                  │ Filterable:      │                       │
│                  │   language, year,│                       │
│                  │   subjects,      │                       │
│                  │   entities       │                       │
│                  └──────────────────┘                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE                            │
│                                                             │
│  User Query                                                 │
│      │                                                      │
│      ▼                                                      │
│  Query Understanding                                        │
│      ├─ Spell correction (SymSpell / dictionary-based)      │
│      ├─ Intent classification                               │
│      │   (keyword / semantic / author / "books like X")     │
│      └─ Query expansion (synonyms, related terms)           │
│      │                                                      │
│      ▼                                                      │
│  Cache Check (Redis)                                        │
│      ├─ HIT  → return cached results                        │
│      └─ MISS ↓                                              │
│      │                                                      │
│      ▼                                                      │
│  Embed query (same model, real-time)                        │
│      │                                                      │
│      ▼                                                      │
│  Azure AI Search (hybrid: BM25 + vector, RRF fusion)        │
│      │                                                      │
│      ▼  top-50 candidates                                   │
│  Cross-Encoder Reranker (PyTorch, CPU)                      │
│      │  ms-marco-MiniLM-L-6-v2                              │
│      │                                                      │
│      ▼  top-10 reranked                                     │
│      │                                                      │
│      ├──► /search  → return results + cache write           │
│      │                                                      │
│      └──► /ask     → RAG: results + LLM → natural language  │
│           │          answer with citations (Azure OpenAI)    │
│           │                                                  │
│      ▼                                                      │
│  Log query + clicks (Cosmos DB) → feedback loop → eval      │
│      │                                                      │
│      ▼                                                      │
│  Client (minimal frontend / API consumers)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Data Ingestion & ETL

**Purpose:** Download, clean, and normalize OpenLibrary bulk data.

```
src/
  etl/
    download.py       # fetch OL dumps from archive.org
    parse.py          # stream-parse JSON dumps (line-delimited)
    clean.py          # normalize text, deduplicate, filter language
    schema.py         # Pydantic models for Book, Author, Edition
    pipeline.py       # orchestrate: download → parse → clean → upload
```

**Key decisions:**
- Stream-parse with `orjson` — dumps are multi-GB, can't load into memory
- Filter to English books with descriptions (cuts 30M → ~500K usable)
- Combine works + editions: use work-level metadata, edition-level ISBNs
- Output: cleaned JSONL → Azure Blob Storage

**Fields to extract:**
```python
class Book(BaseModel):
    work_id: str              # OpenLibrary work key
    title: str
    authors: list[str]
    description: str | None   # first 512 chars
    subjects: list[str]       # top 10 subjects
    first_publish_year: int | None
    cover_id: int | None      # for thumbnail URL
    isbn: list[str]
    language: str
    ratings_average: float | None
```

### 2. Document Enrichment

**Purpose:** Enhance raw book metadata at index time to improve search quality.

```
src/
  enrichment/
    entities.py       # extract people, places, time periods from descriptions
    subjects.py       # normalize & deduplicate OL's messy subject tags
    description.py    # generate descriptions for books missing them (LLM batch)
    pipeline.py       # orchestrate enrichment steps
```

**What gets enriched:**

| Field | Source | Why |
|---|---|---|
| `normalized_subjects` | OL subjects → mapped to controlled vocab | OL has "Fiction", "fiction", "FICTION", "Literary fiction" all separate |
| `entities` | spaCy NER on descriptions | Enables "books set in Paris" or "books about WWII" filters |
| `generated_description` | Azure OpenAI (batch, for books with no description) | ~40% of OL works lack descriptions — huge recall gap |
| `reading_level` | Heuristic (word count, sentence complexity) | Useful filter, easy to compute |

**Cost note:** LLM-generated descriptions use Azure OpenAI batch API (50% cheaper)
on only the ~200K books missing descriptions. One-time cost ~$5-10.

### 3. Embedding Pipeline

**Purpose:** Generate vector embeddings for corpus + real-time queries.

```
src/
  embeddings/
    model.py          # load/export sentence-transformer
    batch.py          # batch embed corpus (offline, CPU VM)
    serve.py          # real-time query embedding (Container App)
    onnx_export.py    # export to ONNX for faster CPU inference
```

**Model:** `sentence-transformers/all-MiniLM-L6-v2`
- 384 dimensions, 22M params, fast on CPU
- ONNX export for ~2x CPU speedup

**Batch pipeline:**
1. Read cleaned JSONL from Blob Storage
2. Concatenate: `f"{title} by {authors}. {description}. {subjects}"`
3. Encode in batches of 256
4. Write embeddings alongside doc metadata
5. Push to AI Search index

**Serving (real-time query embedding):**
- Azure Container App, scale-to-zero
- ONNX Runtime for inference
- Single query latency target: <50ms on CPU

### 4. Azure AI Search Index

**Purpose:** Hybrid retrieval (BM25 + vector) with built-in RRF.

**Index schema:**
```json
{
  "fields": [
    {"name": "id",           "type": "Edm.String",  "key": true},
    {"name": "title",        "type": "Edm.String",  "searchable": true, "analyzer": "en.lucene"},
    {"name": "authors",      "type": "Edm.String",  "searchable": true},
    {"name": "description",  "type": "Edm.String",  "searchable": true, "analyzer": "en.lucene"},
    {"name": "subjects",     "type": "Collection(Edm.String)", "filterable": true, "facetable": true},
    {"name": "year",         "type": "Edm.Int32",   "filterable": true, "sortable": true},
    {"name": "language",     "type": "Edm.String",  "filterable": true},
    {"name": "cover_url",    "type": "Edm.String",  "retrievable": true},
    {"name": "isbn",         "type": "Collection(Edm.String)", "filterable": true},
    {"name": "embedding",    "type": "Collection(Edm.Single)", "dimensions": 384,
     "vectorSearchProfile": "hnsw-profile"}
  ]
}
```

**Search flow:**
1. Hybrid query: BM25 on text fields + kNN on embedding field
2. AI Search fuses with Reciprocal Rank Fusion (built-in)
3. Return top-50 candidates to reranker

### 5. Cross-Encoder Reranker (PyTorch)

**Purpose:** Rerank AI Search candidates with a more powerful model.

```
src/
  reranker/
    model.py          # load cross-encoder (PyTorch / ONNX)
    serve.py          # FastAPI endpoint for reranking
    evaluate.py       # compare with/without reranker
```

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Takes (query, document) pairs, outputs relevance score
- ~22M params, runs on CPU in <500ms for 50 pairs

**Reranking logic:**
```python
def rerank(query: str, candidates: list[Book], top_k: int = 10) -> list[Book]:
    pairs = [(query, f"{b.title} {b.description}") for b in candidates]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [book for book, score in ranked[:top_k]]
```

**Deployment:** Azure Container App (CPU, scale-to-zero)

### 6. Query Understanding

**Purpose:** Process raw user queries before retrieval to improve result quality.

```
src/
  query/
    spell.py          # spell correction (SymSpell with book-domain dictionary)
    intent.py         # classify: keyword | semantic | author | similar-to
    expand.py         # query expansion (synonyms, related terms)
    pipeline.py       # orchestrate: correct → classify → expand
```

**Intent classification:**

Lightweight rule-based + small classifier approach (no GPU needed):

| Query Pattern | Detected Intent | Search Behavior |
|---|---|---|
| `"dune frank herbert"` | `author_title` | Boost BM25 weight, exact match on author |
| `"books about loneliness in space"` | `semantic` | Boost vector weight |
| `"books like Project Hail Mary"` | `similar_to` | Find book → use its embedding as query vector |
| `"978-0-13-468599-1"` | `isbn_lookup` | Direct filter, skip vector search |
| `"scifi publshed 2020"` | `keyword` (with typo) | Spell-correct → `"scifi published 2020"` |

**Spell correction:**
- SymSpell (symmetric delete) with a custom dictionary built from corpus
  titles, authors, and subjects during indexing
- Zero external API cost, <1ms latency
- Falls back to Levenshtein distance for unknown tokens

**Query expansion:**
- Synonym mapping for common book genres (e.g., "sci-fi" → "science fiction")
- Not aggressive — only high-confidence expansions to avoid topic drift

### 7. Caching Layer

**Purpose:** Reduce latency and Azure AI Search costs for repeated queries.

```
src/
  cache/
    manager.py        # cache get/set/invalidate logic
    keys.py           # deterministic cache key generation
    config.py         # TTL, max size, eviction policy
```

**Strategy:**

```
┌─────────────────────────────────────────────┐
│            Cache Architecture               │
│                                             │
│  L1: In-process LRU (100 queries)           │
│      → <1ms, free, per-instance             │
│                                             │
│  L2: Redis (Azure Cache for Redis, Basic)   │
│      → <5ms, shared across instances        │
│      → TTL: 1hr for search results          │
│      → TTL: 24hr for embeddings             │
│                                             │
│  Cache key = hash(normalized_query +        │
│                   filters + intent)          │
└─────────────────────────────────────────────┘
```

**What gets cached:**
| Item | TTL | Why |
|---|---|---|
| Query embeddings | 24h | Same query = same vector, expensive to recompute |
| AI Search results (pre-rerank) | 1h | Most expensive call in the pipeline |
| Final reranked results | 1h | Full pipeline result |
| Spell corrections | 7d | Dictionary-based, very stable |

**Budget note:** Azure Cache for Redis Basic (250MB) = ~$16/mo.
Alternative: skip Redis, use only L1 in-process cache for the prototype.
L1 alone still eliminates the most common repeated queries.

### 8. RAG Answer Endpoint

**Purpose:** Generate natural language answers grounded in search results.

```
src/
  rag/
    prompt.py         # prompt templates for different intents
    generate.py       # call Azure OpenAI with retrieved context
    citations.py      # map answer spans back to source books
    guardrails.py     # hallucination checks, answer validation
```

**Endpoint:** `POST /ask`

```json
// Request
{
  "question": "What are the best books about the history of computing?",
  "max_sources": 5
}

// Response
{
  "answer": "Several well-regarded books cover computing history from different
             angles. **The Innovators** by Walter Isaacson traces the people
             behind the digital revolution [1]. For early computing,
             **The Information** by James Gleick covers the theoretical
             foundations [2]. **Hackers** by Steven Levy captures the culture
             of early programmers [3]...",
  "sources": [
    {"rank": 1, "title": "The Innovators", "author": "Walter Isaacson", "work_id": "OL123W"},
    {"rank": 2, "title": "The Information", "author": "James Gleick", "work_id": "OL456W"},
    {"rank": 3, "title": "Hackers", "author": "Steven Levy", "work_id": "OL789W"}
  ],
  "search_metadata": {
    "retrieval_strategy": "hybrid_reranked",
    "candidates_considered": 50,
    "latency_ms": {"retrieval": 120, "rerank": 200, "generation": 800}
  }
}
```

**Pipeline:**
1. Run the full search pipeline (query understanding → hybrid search → rerank)
2. Build context from top-K reranked results (title, author, description)
3. Call Azure OpenAI (`gpt-5.4-nano` — newest nano model, excellent reasoning, 1M+ context)
4. Post-process: extract citations, validate answer references real books
5. Return answer + sources + latency breakdown

**Prompt design:**
```python
SYSTEM_PROMPT = """You are a book recommendation assistant. Answer the user's
question using ONLY the book information provided below. Cite books by their
[number]. If the provided books don't answer the question, say so honestly.
Do not invent books or facts."""

CONTEXT_TEMPLATE = """
[{rank}] "{title}" by {authors} ({year})
{description}
Subjects: {subjects}
"""
```

**Guardrails:**
- Every book title in the answer must appear in the source list
- If the model hallucinates a title, strip it and flag in response metadata
- Token budget: max 500 tokens generation (cost control)

**Cost:** `gpt-5.4-nano` at ~$0.20/1M input, $1.25/1M output — still essentially free at search volumes (~$0.0003/query).

### Model selection rationale

| Model | $/query (est.) | Quality | Pick when |
|---|---|---|---|
| `gpt-5.4-nano` | ~$0.0003 | Best nano-class reasoning (2026) | Default — quality + budget |
| `gpt-4.1-nano` | ~$0.00007 | Good for simple grounded answers | Ultra-budget fallback |
| `gpt-5.4-mini` | ~$0.001 | Strong reasoning, high quality | Complex multi-hop queries |

Start with `gpt-5.4-nano`, fall back to `gpt-4.1-nano` only if cost becomes a concern at scale.

### 9. API Layer

**Purpose:** Unified search API with observability.

```
src/
  api/
    main.py           # FastAPI app
    routes/
      search.py       # /search endpoint
      health.py       # /health, /ready
    middleware/
      logging.py      # structured logging
      metrics.py      # latency, result quality tracking
    models/
      request.py      # SearchRequest schema
      response.py     # SearchResponse schema
```

**Endpoints:**
```
GET  /search?q=...&filters=...&page=...&explain=true
GET  /search/compare?q=...          # side-by-side: BM25 vs hybrid vs reranked
POST /ask                            # RAG: natural language answer + citations
GET  /suggest?q=...                  # autocomplete / spell suggestions
GET  /health
GET  /metrics                       # Prometheus format
```

**`/search/compare`** is a portfolio differentiator — lets you visually demo
why hybrid + reranking beats keyword search.

### 10. Evaluation Framework

**Purpose:** Measure and demonstrate search quality improvements.

```
src/
  eval/
    dataset.py        # build eval set from OL ratings/lists
    metrics.py        # MRR, NDCG@10, Precision@K
    compare.py        # compare retrieval strategies
    report.py         # generate eval report (markdown/HTML)
```

**Evaluation dataset sources:**
- OpenLibrary reading lists (user-curated → relevance signal)
- Synthetic: GPT-generated query–book pairs
- Manual: 50-100 hand-labeled queries

**Strategies to compare:**
| Strategy | Description |
|---|---|
| BM25 only | text_query → AI Search, keyword mode |
| Vector only | embed_query → AI Search, vector mode |
| Hybrid (RRF) | both → AI Search, hybrid mode |
| Hybrid + Reranker | hybrid top-50 → cross-encoder → top-10 |

---

## Infrastructure

### Azure Resources

```
Resource Group: rg-book-search
├── Azure AI Search (Basic, $75/mo)
├── Azure Container Apps Environment
│   ├── app-api          # FastAPI search service
│   ├── app-embedder     # query embedding service
│   └── app-reranker     # cross-encoder reranker
├── Azure Cache for Redis (Basic, 250MB)  # query + embedding cache
├── Azure Blob Storage   # raw dumps + cleaned corpus
├── Azure Container Registry
├── Azure Cosmos DB (serverless)  # query logs, click logs, eval results
├── Azure OpenAI (gpt-5.4-nano)   # RAG answer generation + LLM-as-judge
└── Azure Monitor / Log Analytics
```

### IaC (Bicep or Terraform)

```
infra/
  main.bicep            # all Azure resources
  modules/
    search.bicep
    container-apps.bicep
    storage.bicep
    monitoring.bicep
  parameters/
    dev.bicepparam
    prod.bicepparam
```

### CI/CD (GitHub Actions)

```yaml
# .github/workflows/deploy.yml
# Triggers: push to main
# Steps:
#   1. Lint + test
#   2. Build container images
#   3. Push to ACR
#   4. Deploy infra (Bicep)
#   5. Deploy apps (Container Apps)
#   6. Run smoke tests

# .github/workflows/reindex.yml
# Triggers: manual / scheduled (weekly)
# Steps:
#   1. Download latest OL dumps
#   2. Run ETL pipeline
#   3. Batch embed
#   4. Push to AI Search index
#   5. Run eval suite, post results to PR
```

---

## Budget ($150/mo VS Enterprise credits)

| Resource | SKU | Est. Cost |
|---|---|---|
| AI Search | Basic (1 unit) | $75 |
| Container Apps | Consumption (scale-to-zero) | $10-15 |
| Redis Cache | Basic C0 (250MB) | $16 |
| Blob Storage | LRS, ~10GB | $1 |
| Container Registry | Basic | $5 |
| Cosmos DB | Serverless | $5-10 |
| Azure OpenAI | gpt-5.4-nano (pay-per-token) | ~$1 |
| Log Analytics | Free tier (5GB/mo) | $0 |
| Batch VM (one-time) | D4s_v3, ~1hr | $0.20 |
| **Total** | | **~$115-125/mo** |

**Budget optimization tips:**
- Skip Redis initially — use in-process LRU cache for Phase 0-1
- Cosmos DB serverless has no idle cost — only pay for actual RUs
- Azure OpenAI for RAG is negligible (~$0.001/query with gpt-4o-mini)
- Tear down batch VMs immediately after embedding jobs complete

---

## Development Phases

### Phase 0 — Prototype (1 week)
- [ ] EDA notebook on small dataset (UCSD Book Graph or OL subset)
- [ ] Local FastAPI + FAISS proof-of-concept
- [ ] Validate embedding quality on sample queries
- [ ] No Azure, no infra — just prove the search works

### Phase 1 — Core Pipeline (2 weeks)
- [ ] ETL pipeline for OpenLibrary dumps
- [ ] Document enrichment (subject normalization, entity extraction)
- [ ] Batch embedding pipeline
- [ ] Azure AI Search index setup + data push
- [ ] Hybrid search API (FastAPI)
- [ ] Basic /search endpoint working end-to-end on Azure

### Phase 2 — Reranker + Query Intelligence (1-2 weeks)
- [ ] Cross-encoder reranker (PyTorch, ONNX export)
- [ ] Deploy reranker on Container App
- [ ] Query understanding (spell correction, intent classification)
- [ ] Caching layer (in-process LRU → Redis)
- [ ] Build eval dataset
- [ ] Evaluation framework (MRR, NDCG)
- [ ] /search/compare endpoint

### Phase 3 — RAG + Production Polish (1-2 weeks)
- [ ] RAG /ask endpoint (Azure OpenAI gpt-4o-mini)
- [ ] Citation extraction + hallucination guardrails
- [ ] Infrastructure-as-Code (Bicep)
- [ ] CI/CD pipelines
- [ ] Monitoring + dashboards
- [ ] Query + click logging (Cosmos DB)

### Phase 4 — Documentation + Demo (1 week)
- [ ] README with architecture diagram
- [ ] Eval results report (MRR/NDCG charts per strategy)
- [ ] Demo video / write-up
- [ ] Blog post explaining design decisions

### Phase 5 — Stretch Goals
- [ ] **Learned sparse retrieval (SPLADE)** — PyTorch model replacing BM25 with learned term weights, ONNX-exported for CPU serving
- [ ] **ColBERT late-interaction retrieval** — token-level similarity as alternative to single-vector
- [ ] **Distillation pipeline** — train fast bi-encoder using cross-encoder as teacher
- [ ] Fine-tune embedding model on book pairs (contrastive learning on click data)
- [ ] LLM-generated descriptions for books missing them
- [ ] User feedback loop (click signals → eval dataset)
- [ ] Scale to full OL dump (5M+ records)
- [ ] A/B testing framework for retrieval strategies

---

## Key Portfolio Talking Points

1. **Data engineering** — streamed multi-GB dumps, cleaned messy real-world data, enriched with NER + LLM
2. **ML infra** — deployed PyTorch models (embedding + reranker) on CPU, ONNX optimized
3. **Search engineering** — hybrid retrieval, RRF fusion, cross-encoder reranking, query understanding
4. **RAG** — grounded answer generation with citation tracking and hallucination guardrails
5. **Evaluation** — quantitative proof that each layer improves search quality (MRR/NDCG)
6. **Cloud infra** — IaC, CI/CD, scale-to-zero, cost-optimized for $150/mo budget
7. **Production mindset** — caching, observability, health checks, structured logging, query analytics
