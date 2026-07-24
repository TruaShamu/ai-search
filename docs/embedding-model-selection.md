# Embedding Model Selection — Decision Document

## Context

We need an embedding model for a book search engine over ~250K–1.6M OpenLibrary
records. The model must:
- Run on **CPU** (no GPU budget)
- Be **open-source** (deployable on Azure Container Apps or Foundry)
- Handle short-to-medium text (median embedding input ~86 chars, max ~1000)
- Produce good **retrieval** quality (not just STS similarity)
- Be affordable to batch-embed ~250K–1.6M documents

---

## Candidates

### 1. all-MiniLM-L6-v2 (sentence-transformers)

| Spec | Value |
|---|---|
| Parameters | 22M |
| Dimensions | 384 |
| Max tokens | 256 |
| MTEB Retrieval (nDCG@10) | ~47–50 |
| CPU speed (batch) | ~500 docs/sec |
| Model size | ~80 MB |

**Pros:**
- Tiny, blazing fast on CPU — embed 250K docs in ~8 minutes
- Battle-tested, massive community adoption
- Trivial to deploy, minimal memory

**Cons:**
- Lowest retrieval quality of all candidates (~13 points behind SOTA)
- 256 token limit truncates longer descriptions
- No Matryoshka dimension support
- Getting old (2021 model)

**Verdict:** Good for prototyping, but leaving quality on the table for production.

---

### 2. BGE-M3 (BAAI)

| Spec | Value |
|---|---|
| Parameters | 567M |
| Dimensions | 1024 |
| Max tokens | 8192 |
| MTEB Retrieval (nDCG@10) | ~60–61 |
| CPU speed (batch) | ~30–60 docs/sec |
| Model size | ~2.2 GB |

**Pros:**
- Hybrid output: dense + sparse + ColBERT multi-vector (all in one model)
- Excellent multilingual support (100+ languages)
- High retrieval quality, proven in production RAG systems
- Long context window (8192 tokens)

**Cons:**
- 25x larger than MiniLM — slower on CPU, ~2.2 GB memory
- Batch embedding 250K docs on CPU: ~1–2 hours (vs. 8 min for MiniLM)
- 1024-dim vectors = ~2.7x more storage in vector index
- Hybrid output is powerful but adds complexity (do we need sparse + dense?)

**Verdict:** Overkill for our use case. Hybrid features overlap with AI Search's
built-in BM25. Storage cost of 1024-dim vectors matters at scale.

---

### 3. jina-embeddings-v5-text-nano (Jina AI)

| Spec | Value |
|---|---|
| Parameters | 239M |
| Dimensions | 768 (Matryoshka: 32–768) |
| Max tokens | 8192 |
| MTEB Retrieval (nDCG@10) | ~65 (SOTA for <500M params) |
| CPU speed (batch) | ~100–200 docs/sec |
| Model size | ~900 MB |

**Pros:**
- Best retrieval quality per parameter of any model in this class
- Matryoshka dimensions — can use 256d or 384d for faster search with minor quality loss
- 8192 token context (future-proof for longer descriptions)
- 32-language support, multilingual-aware
- <5ms per query on CPU — great for real-time query embedding
- 2026 model — latest architecture and training

**Cons:**
- 10x larger than MiniLM (but 2.5x smaller than BGE-M3)
- Batch embedding 250K on CPU: ~20–40 minutes
- Newer model, less community battle-testing than MiniLM or BGE
- Jina licensing (Apache 2.0 but verify for commercial)

**Verdict:** Best balance of quality, size, and speed for our use case.

---

### 4. nomic-embed-text-v1.5 (Nomic AI)

| Spec | Value |
|---|---|
| Parameters | 137M |
| Dimensions | 768 (Matryoshka: 64–768) |
| Max tokens | 8192 |
| MTEB Retrieval (nDCG@10) | ~62 |
| CPU speed (batch) | ~200–400 docs/sec |
| Model size | ~274 MB |

**Pros:**
- Excellent size/quality ratio — 137M params, only 274 MB
- Matryoshka dimensions (64–768) — same flexibility as Jina
- 8192 token context
- Fully open source (Apache 2.0), no licensing concerns
- Very fast on CPU — nearly as fast as MiniLM, far better quality
- Well-documented, strong community adoption
- ONNX support out of the box

**Cons:**
- ~3 points below Jina nano on retrieval (62 vs 65)
- Primarily English-focused (not ideal if multilingual matters later)
- No vision/multimodal variant in v1.5 (v2+ adds this)

**Verdict:** The sweet spot between MiniLM and Jina. Almost as fast as MiniLM,
almost as good as Jina, and smaller than both BGE-M3 and Harrier. Very strong
contender.

---

### 5. harrier-oss-v1-0.6b (Microsoft)

| Spec | Value |
|---|---|
| Parameters | 600M |
| Dimensions | 1024 |
| Max tokens | 8192 |
| MTEB Retrieval (nDCG@10) | ~60–62 |
| CPU speed (batch) | ~30–50 docs/sec |
| Model size | ~2.3 GB |

**Pros:**
- Microsoft's own model — first-class support on Azure Foundry
- MIT license, truly open source
- Excellent multilingual (94 languages)
- Strong on legal/medical/domain retrieval
- Politically safe choice on Azure (vendor alignment)

**Cons:**
- Similar size to BGE-M3 but slightly lower retrieval scores
- Slower on CPU than Jina nano
- 1024-dim vectors (same storage concern as BGE-M3)
- Newer model, less benchmarked outside Microsoft's own evals

**Verdict:** Strong if you want vendor alignment with Azure. But Jina nano beats
it on quality-per-parameter and is more CPU-friendly.

---

### 6. Azure OpenAI text-embedding-3-small (API option)

| Spec | Value |
|---|---|
| Parameters | Unknown (proprietary) |
| Dimensions | 1536 (or 256/512/1024 via truncation) |
| Max tokens | 8191 |
| MTEB Retrieval (nDCG@10) | ~64 |
| Speed | API-limited (~1000 RPM) |
| Cost | ~$0.02 / 1M tokens |

**Pros:**
- Zero infra to manage — just an API call
- High quality retrieval
- Dimension flexibility via truncation
- Batch API available (50% cheaper)

**Cons:**
- Proprietary — can't show model deployment/ONNX/infra skills in portfolio
- API rate limits slow down batch embedding
- Per-token cost adds up: 250K docs x ~50 tokens avg = ~$0.25 (negligible)
- No PyTorch involvement — weakens the portfolio story

**Verdict:** Easiest path but defeats the portfolio purpose. Could use as a
comparison baseline in eval.

---

## Comparison Matrix

| | MiniLM | Nomic v1.5 | Jina v5 nano | BGE-M3 | Harrier 0.6b | AzOAI 3-small |
|---|---|---|---|---|---|---|
| **Retrieval quality** | ★★ (47) | ★★★★ (62) | ★★★★★ (65) | ★★★★ (61) | ★★★★ (61) | ★★★★ (64) |
| **CPU speed** | ★★★★★ | ★★★★★ | ★★★★ | ★★ | ★★ | N/A (API) |
| **Model size** | 80MB | 274MB | 900MB | 2.2GB | 2.3GB | N/A |
| **Portfolio value** | ★★★ | ★★★★ | ★★★★★ | ★★★★ | ★★★★ | ★ |
| **Azure Foundry fit** | ★★★ | ★★★ | ★★★★ | ★★★ | ★★★★★ | ★★★★★ |
| **Matryoshka dims** | No | Yes (64-768) | Yes (32-768) | No | No | Yes |
| **Max tokens** | 256 | 8192 | 8192 | 8192 | 8192 | 8191 |
| Embed 250K (CPU) | ~8 min | ~10-15 min | ~20-40 min | ~1-2 hr | ~1-2 hr | ~25 min (API) |

---

## Recommendation

### Primary: jina-embeddings-v5-text-nano

**Why:**
1. Best retrieval quality for a sub-250M model (MTEB nDCG@10 ~65)
2. Matryoshka dims let us use 384d for prototype, upgrade to 768d later (same model!)
3. Fast enough on CPU for both batch embedding and real-time queries
4. 8192 token context future-proofs for longer descriptions
5. Strong portfolio story: "I evaluated 6 models, chose the optimal quality/cost tradeoff"
6. ONNX-exportable for production CPU inference

### Strong alternative: nomic-embed-text-v1.5

Nearly as good at retrieval (62 vs 65), but **3x smaller** (274MB vs 900MB) and
**2x faster on CPU**. If deployment size or batch speed matters more than
squeezing the last 3 MTEB points, Nomic is the better pick. Also has Matryoshka
support and a cleaner Apache 2.0 license.

**The honest tradeoff:** Nomic is the pragmatic choice. Jina is the quality-maximizing
choice. Either is defensible — pick based on whether you want to optimize for
speed/simplicity or retrieval quality.

### Fallback: all-MiniLM-L6-v2

Use for Phase 0 prototyping only — it's fast, familiar, and lets you validate
the pipeline before committing to Jina. Then swap in Jina for Phase 1+ and
compare quality in your eval framework.

### Eval baseline: Azure OpenAI text-embedding-3-small

Include in your /search/compare evaluation to show how your open-source model
compares to a proprietary API. Great portfolio talking point.

---

## Suggested Approach

```
Phase 0: Prototype with all-MiniLM-L6-v2 (fast iteration)
Phase 1: Switch to jina-embeddings-v5-text-nano at 384d (Matryoshka)
Phase 2: Eval — compare MiniLM vs Jina-384d vs Jina-768d vs Azure OpenAI
Phase 3: Publish eval results showing quality/cost tradeoffs
```

This gives you a compelling narrative:
"I started with a lightweight model, upgraded based on retrieval metrics,
and quantified the quality improvement at each step."
