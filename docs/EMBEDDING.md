# Embedding Design

How books are turned into vectors, and why the model, template and dimensionality
were chosen the way they were. The authoritative source for the template is
`src/search/embed.py:build_embedding_texts` — what follows is the reasoning behind
it.

---

## Model: nomic-embed-text-v1.5

Six models were evaluated against the requirements of a book search engine: good
retrieval quality on short-to-medium text (median input ~86 chars), CPU-only
inference (no GPU budget), open-source and self-hostable, and fast enough for
real-time search latency.

The short list came down to two:

| | nomic-embed-text-v1.5 | jina-embeddings-v5-text-nano |
|---|---|---|
| MTEB Retrieval nDCG@10 | ~62 | ~65 |
| Parameters | 137 M | 239 M |
| Model size | 274 MB | 900 MB |
| CPU batch speed | ~200–400 docs/sec | ~100–200 docs/sec |
| Matryoshka | 64 / 128 / 256 / 512 / 768 | 32–768 |
| License | Apache 2.0 | Apache 2.0 |

**Why nomic won.** The 3-point MTEB gap is real but narrow. What decided it:

- **Latency.** Search is interactive — every query embeds in real time on CPU.
  nomic is ~2× faster per inference, which directly affects p95 response time.
- **Cost.** GPU compute is expensive and not in scope. On CPU, 274 MB vs 900 MB
  means lower memory cost per replica and faster cold starts on the 30-replica
  embedding job.
- **Matryoshka support.** Both models offer it, but nomic's trained checkpoints
  at 256d hit a sweet spot — half the storage of 512d with minimal retrieval
  loss. This is what makes a CPU-only dense arm viable at 84.8K documents.
- **Hybrid compensates.** The 3-point gap assumes dense retrieval alone. Fused
  with TF-IDF via RRF, the margin shrinks further — measured retrieval quality
  on the live index is in [EVALUATION.md](EVALUATION.md).

The four models rejected earlier: **all-MiniLM-L6-v2** (80 MB, fast, but MTEB ~47
— too low for production), **BGE-M3** (2.2 GB, built-in hybrid overlaps with our
TF-IDF arm), **harrier-oss-v1** (2.3 GB, Azure-aligned but too large for CPU),
and **Azure OpenAI text-embedding-3-small** (API-only — no ONNX export, adds an
external dependency to every query).

---

## Dimensionality: 256

nomic-embed-text-v1.5 was trained with Matryoshka checkpoints at 768, 512, 256,
128, and 64 dimensions. 256 is the middle ground: half the storage of 512 with
minimal quality loss on MTEB, and 3× smaller vectors than the full 768. At 84,801
points × 256 × float32, the dense index is ~82 MB.

After truncation to 256d the vectors are **re-normalized to unit length** so that
cosine similarity stays valid — this is a Matryoshka requirement that is easy to
forget and produces silently degraded results if skipped.

---

## Embedding Template

Every book is embedded as a single vector. The input string is built by
`src/search/embed.py:build_embedding_texts`:

```
search_document: {title} by {authors}. {description[:2000]}. {subjects[:10]}
```

Key decisions:

- **Nomic `search_document:` / `search_query:` prefixes.** The model was trained
  for asymmetric retrieval (short query → long document). Omitting the prefix
  drops retrieval quality measurably on MTEB.

- **Description capped at 2,000 chars.** The model accepts 8,192 tokens but
  longer inputs dilute title and author signal. 2,000 chars covers >99% of
  descriptions without truncation.

- **Subjects included.** Genre tags add retrieval signal for topical queries
  ("science fiction about AI") at negligible cost.

- **Year excluded.** Numbers do not embed well. `first_publish_year` was
  originally planned as an Azure AI Search filterable field; on Qdrant it
  could be a payload filter, but is not currently indexed.

- **Single vector per book.** Multi-vector (ColBERT-style) would improve
  long-description retrieval but multiplies storage and complexity. At the
  current scale, single-vector + reranker is sufficient.

---

## Sparse Arm

The sparse side of hybrid search is a **TF-IDF** vectorizer (not BM25), globally
fit over the entire corpus in `src/indexing/load.py`. "Globally fit" means the
vocabulary and IDF weights are computed once across all 84,801 books, and the
same fitted `TfidfVectorizer` object is pickled and shipped in the API container
so the query encoder and the index share identical term weights.

BM25's length normalization matters more as the corpus grows — at 84.8K docs this
is the closest design call in the system and the most likely next change.
