# 📚 BookSearch — Hybrid Semantic Search Engine

A hybrid search engine over 26,500+ books from the OpenLibrary catalog. Combines TF-IDF sparse retrieval with dense vector search (nomic-embed-text-v1.5) using Reciprocal Rank Fusion, plus an optional cross-encoder reranker — all self-hosted on Qdrant and deployed to Azure Container Apps with full CI/CD.

Built as a portfolio piece demonstrating **backend + ML infrastructure engineering**.

---

## Architecture

```mermaid
graph TD
    subgraph Frontend
        UI[Next.js + shadcn/ui<br/>Search · Compare · Ask · Rerank Toggle]
    end

    subgraph API Layer
        FP[FastAPI]
        QU[Query Understanding<br/>Spell Correct · Intent Routing]
        RR[Cross-Encoder Reranker<br/>ONNX · ms-marco-MiniLM]
        RAG[RAG Pipeline<br/>GPT + Citation Validation]
    end

    subgraph Vector DB
        QD[Qdrant · 26,519 points]
        DV[Dense: nomic-embed-text-v1.5<br/>dim=256, Matryoshka]
        SV[Sparse: TF-IDF vectors]
        RRF[RRF Fusion]
    end

    subgraph Data Pipeline
        OL[OpenLibrary Dump<br/>250K works]
        GR[Goodreads Augmentation<br/>+13K descriptions]
        EMB[Cloud Embedding<br/>ACI · 4 CPU · 16GB]
        MIG[Migration<br/>FAISS → Qdrant]
    end

    UI -->|HTTP| FP
    FP --> QU --> QD
    QD --> DV & SV
    DV & SV --> RRF --> FP
    FP -.->|opt-in| RR --> FP
    FP --> RAG

    OL --> GR --> EMB --> MIG --> QD
```

### Query Flow

```mermaid
sequenceDiagram
    participant User
    participant API as FastAPI
    participant QU as Query Understanding
    participant Q as Qdrant
    participant RR as Reranker (opt-in)

    User->>API: GET /search?q=love+story+tragedy&rerank=true
    API->>QU: Spell correct + intent detect
    QU-->>API: corrected query, mode=hybrid
    API->>Q: Prefetch dense (top 50) + sparse (top 50)
    Q-->>API: RRF fused results (top 50)
    API->>RR: Score 50 candidates
    RR-->>API: Reranked top 10
    API-->>User: Romeo & Juliet #1, Carmen #2, ...
```

### Data Pipeline

```mermaid
flowchart LR
    A[OpenLibrary Dump<br/>250K works] -->|Filter: has description| B[13.4K Tier 1]
    A -->|No description| C[237K Tier 2]
    C -->|Title-match join| D[Goodreads HF Dataset<br/>~1M books]
    D -->|+13K matched| E[26,519 books]
    B --> E
    E -->|nomic-embed-text-v1.5<br/>dim=256| F[Dense Vectors]
    E -->|TfidfVectorizer| G[Sparse Vectors]
    F & G -->|migrate.py| H[(Qdrant)]
```

### Embedding Worker (Event-Driven)

```mermaid
sequenceDiagram
    participant ETL as Enqueue Script
    participant Q as Azure Storage Queue
    participant W as ACA Job (scale 0→1)
    participant Blob as Azure Blob Storage
    participant QD as Qdrant

    ETL->>Q: Enqueue N batch messages<br/>(start_idx, end_idx, blob_path)
    Note over W: Idle (0 replicas)
    Q-->>W: KEDA trigger (message arrives)
    W->>Blob: Download book slice [start, end)
    W->>W: Embed with nomic-embed-text-v1.5
    W->>QD: Upsert dense + sparse vectors
    W->>Q: Delete message (ack)
    Note over W: Scale back to 0
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Next.js 16, TypeScript, Tailwind CSS, shadcn/ui |
| **API** | FastAPI, Python 3.11, Pydantic |
| **Vector DB** | Qdrant (self-hosted, Docker) |
| **Embeddings** | nomic-embed-text-v1.5 (Matryoshka, dim=256) |
| **Sparse Retrieval** | TF-IDF (scikit-learn) |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 (ONNX Runtime) |
| **RAG** | Azure OpenAI (GPT) with citation validation |
| **Query Understanding** | SymSpell (spell correction) + intent routing |
| **Cloud Infra** | Azure Container Instances, ACR, Blob Storage |
| **IaC** | Bicep (ACA templates ready) |

---

## Features

- **Hybrid Search (RRF)** — Reciprocal Rank Fusion of TF-IDF + dense vectors. Best of both worlds: keyword precision + semantic understanding.
- **Cross-Encoder Reranker** — Optional two-stage retrieval. ONNX-optimized for CPU. Available as a toggle — see eval findings below on when it helps vs. hurts.
- **Query Understanding** — Spell correction (SymSpell), intent detection, query-adaptive mode routing.
- **RAG with Guardrails** — Natural language Q&A grounded in retrieved books. Citation validation prevents hallucinated titles.
- **Compare View** — Side-by-side 3-column comparison of keyword vs. hybrid vs. vector results.
- **Evaluation Framework** — Reproducible eval harness computing MRR@10, NDCG@10, Recall@10 against a 30-query labeled dataset. Run via `scripts/eval_via_api.py`.

---

## Eval Results

Evaluated on a 30-query labeled dataset (LLM-judged relevance, graded 0/1/2) against the live production deployment. Reproducible via `python scripts/eval_via_api.py`.

### Retrieval Quality (k=10, 26.5K corpus)

| Mode | MRR@10 | NDCG@10 | Recall@10 | Avg Latency |
|------|--------|---------|-----------|-------------|
| Keyword (TF-IDF) | 0.662 | 0.354 | 0.282 | 73ms |
| Vector (nomic-256d) | 0.625 | 0.295 | 0.205 | 45ms |
| **Hybrid (RRF)** | **0.665** | **0.385** | **0.306** | 76ms |
| Hybrid + Rerank | 0.611 | 0.370 | 0.273 | 4340ms |

### Key Findings

- **RRF fusion (hybrid) is the clear winner** — +6% NDCG and +10% recall over keyword-only, with negligible latency cost (76ms vs 73ms).
- **Cross-encoder reranking hurts on this dataset.** MRR drops 0.665 → 0.611, recall drops 0.306 → 0.273, and latency balloons to 4.3s. The likely cause: `ms-marco-MiniLM-L-6-v2` was trained on web passage retrieval, not book metadata. Short book titles + subjects don't give the cross-encoder enough signal to improve over RRF's rank fusion.
- **Vector alone underperforms keyword** — at this corpus size, TF-IDF captures title/author matches that dense embeddings miss. Hybrid is needed to combine both strengths.

> The reranker still adds value for exploratory/natural-language queries where the user's intent doesn't match exact keywords (e.g., "love story tragedy" → Romeo & Juliet rises from rank 4 to rank 1). It's exposed as an opt-in toggle in the UI.

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker (for Qdrant)
- Node.js 18+ / pnpm (for frontend)

### Quick Start

```bash
# 1. Clone and install
git clone https://github.com/TruaShamu/ai-search.git
cd ai-search
pip install -r requirements.txt

# 2. Start Qdrant
docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant:latest

# 3. Run migration (loads data into Qdrant)
python -m src.qdrant.migrate --qdrant-url http://localhost:6333 --collection books --recreate

# 4. Start API
export QDRANT_URL=http://localhost:6333
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 5. Start frontend
cd web && pnpm install && pnpm dev
```

Open http://localhost:3000

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `QDRANT_COLLECTION` | Collection name | `books` |
| `AZURE_OPENAI_ENDPOINT` | For RAG /ask endpoint | — |
| `AZURE_OPENAI_API_KEY` | For RAG /ask endpoint | — |

---

## Project Structure

```
src/
├── api/            FastAPI application (search, ask, health, stats)
├── qdrant/         Qdrant client (hybrid search) + migration script
├── search/         Embedding pipeline (nomic-embed-text-v1.5)
├── reranker/       Cross-encoder reranker (ONNX + PyTorch fallback)
├── rag/            RAG generation with hallucination guardrails
├── query/          Query understanding (spell, intent, expansion)
├── eval/           Evaluation framework (MRR, NDCG, Recall)
├── etl/            Data pipelines (OpenLibrary + Goodreads augmentation)
└── azure_search/   Azure AI Search client (legacy, superseded by Qdrant)

web/                Next.js frontend (search UI, compare view, ask tab)
infra/              Bicep templates (ACA deployment)
scripts/            Cloud embedding automation
data/
├── raw/            OpenLibrary dumps
├── processed/      Augmented catalog (books_augmented.jsonl)
├── index/          FAISS index, metadata, TF-IDF vectorizer
├── eval/           Evaluation datasets + results
└── models/         ONNX reranker model
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Qdrant over Azure AI Search** | 15x faster (24ms vs 370ms), no tier limits, built-in RRF, self-hosted |
| **TF-IDF over BM25** | Sufficient at 26K scale; hybrid compensates. BM25's length norm matters more at >100K docs |
| **Matryoshka dim=256** | nomic-embed-text-v1.5 trained checkpoints: 768/512/256/128/64. 256 balances quality vs. index size |
| **Reranker opt-in** | Adds latency; helps exploratory queries but hurts on keyword-heavy ones (see eval). User's choice via toggle |
| **Goodreads augmentation** | OpenLibrary lacks descriptions for 95% of books. Title-match join doubled the corpus |
| **ONNX reranker** | 3.7x faster than PyTorch on CPU (23ms vs 86ms for 4 candidates) |
| **Cloud embedding (ACI)** | Local GPU unavailable; 4-CPU ACI with 16GB RAM handles 26K docs in ~3h |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search` | GET | Hybrid search with mode, rerank, filters |
| `/search/compare` | GET | Side-by-side results across all modes |
| `/ask` | POST | RAG question answering with citations |
| `/health` | GET | Service health check |
| `/stats` | GET | Index statistics |

### Example

```bash
# Hybrid search
curl "http://localhost:8000/search?q=scottish+romance&mode=hybrid&top_k=5"

# With reranking
curl "http://localhost:8000/search?q=love+story+tragedy&rerank=true"

# RAG question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are some good books about Scottish history?"}'
```

---

## License

MIT
