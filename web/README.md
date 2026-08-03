# Frontend

Next.js client for the book search API. Thin by design — it renders results and exposes
the knobs the API already supports; the retrieval, fusion, reranking and evaluation all
live in [`../src`](../src). Start with the [root README](../README.md) for the system.

**[Live](https://black-grass-0df1c7a0f.7.azurestaticapps.net/)** · Next.js 16 (App Router),
React 19, TypeScript, Tailwind + shadcn/ui.

## Running it

The API does not have to be local. `NEXT_PUBLIC_API_URL` decides what the client talks
to, so you can run the UI on its own against the deployed backend and skip Qdrant and the
~190 MB of index artifacts entirely:

```bash
pnpm install
echo 'NEXT_PUBLIC_API_URL=https://booksearch-api.thankfulstone-e6f7cf40.eastus.azurecontainerapps.io' > .env.local
pnpm dev
```

`.env.local` is gitignored, so a fresh clone with no `.env.local` falls back to
`http://localhost:8000` — which fails silently as a network error if nothing is serving
there. That fallback is the usual cause of an empty page.

## What's here

| Path | Contents |
|---|---|
| `src/app/page.tsx` | The only page. Owns search state, mode, and the rerank / spell-correction toggles |
| `src/components/compare-view.tsx` | Runs keyword, vector and hybrid **in parallel** on one query and shows all three rankings side by side, with per-mode latency and the overlap between result sets |
| `src/components/ask-view.tsx` | RAG answers from `POST /ask`, rendered as markdown with sources |
| `src/components/book-card.tsx`, `search-bar.tsx`, `mode-toggle.tsx` | Result rendering and input |
| `src/lib/api.ts` | Every call to the backend. Typed response shapes live here |
| `src/components/ui/` | Unmodified shadcn/ui primitives |

Compare view is the part worth looking at: the measured gaps between the three modes are
in [the evaluation](../docs/EVALUATION.md), and this puts the same comparison in front of
you on your own query rather than asking you to trust a table.

## Notes

Mode labels must track the backend. The keyword arm is **TF-IDF**, not BM25
(`src/qdrant/migrate.py`) — the two are easy to confuse and the labels here read as a
claim about how retrieval works.

Reranking is a per-query toggle, off by default, because it costs roughly 1.8s against
~215ms for hybrid. First request after an idle period may be slow while the API loads
models; `/ready` gates ingress until they're in memory.

`pnpm build` type-checks and builds; `pnpm lint` runs ESLint. Pushes touching `web/**`
deploy to Azure Static Web Apps via
[`deploy-frontend.yml`](../.github/workflows/deploy-frontend.yml).
