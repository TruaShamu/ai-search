# Embedding Strategy — Decision Document

## What We Embed

### Template (Strategy 4 — Tiered with Nomic task prefixes)

```python
# Tier 1 (has description + subjects)
"search_document: {title} by {author}. {description[:512]}. {subjects}. People: {people}. Places: {places}"

# Tier 2 (subjects only)  
"search_document: {title} by {author}. {subjects}. People: {people}. Places: {places}"

# Query-side
"search_query: books about the history of computing"
```

### Why This Approach

1. **Tiered** — doesn't pad empty fields, maximizes signal per data quality level
2. **Nomic `search_document:` / `search_query:` prefixes** — improves asymmetric retrieval (short query → long doc)
3. **subject_people/places included** — free enrichment already in the data (39% have places, 10% people)
4. **Year excluded from embedding text** — numbers don't embed well; goes in AI Search as a filterable field
5. **Single vector per book** — keeps storage and complexity low
6. **Description capped at 512 chars** — balances signal vs token budget

### What Goes Where

| Field | In Embedding? | In BM25 Index? | As Filter? |
|---|---|---|---|
| title | ✅ | ✅ | |
| authors | ✅ | ✅ | |
| description | ✅ (Tier 1) | ✅ | |
| subjects | ✅ | ✅ | ✅ (faceted) |
| subject_places | ✅ | ✅ | ✅ |
| subject_people | ✅ | ✅ | ✅ |
| subject_times | ✅ | | ✅ |
| first_publish_year | ❌ | | ✅ (sortable) |
| cover_id | ❌ | | (display only) |

### Model Choice

**nomic-embed-text-v1.5** — see `docs/embedding-model-selection.md` for full comparison.

- 137M params, 274MB, Apache 2.0
- 768d with Matryoshka (will use 384d for prototype)
- MTEB Retrieval nDCG@10 ~62
- ~200-400 docs/sec on CPU

### Alternatives Evaluated

| Model | MTEB | Size | Decision |
|---|---|---|---|
| all-MiniLM-L6-v2 | 47 | 80MB | Too low quality for production |
| nomic-embed-text-v1.5 | 62 | 274MB | **Selected** — best speed/quality/size |
| jina-v5-text-nano | 65 | 900MB | Close second — 3pts better but 3x larger |
| BGE-M3 | 61 | 2.2GB | Overkill — hybrid overlaps with AI Search BM25 |
| harrier-oss-v1 | 61 | 2.3GB | Azure-aligned but too large for CPU |
| Azure OpenAI 3-small | 64 | API | No portfolio value, eval baseline only |
