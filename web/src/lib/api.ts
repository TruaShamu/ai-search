const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface BookResult {
  id: string;
  title: string;
  authors: string;
  description: string;
  subjects: string[];
  year: number | null;
  cover_url: string | null;
  work_id: string;
  tier: number;
  score: number;
}

export interface QueryUnderstanding {
  original: string;
  corrected: string;
  was_corrected: boolean;
  intent: string;
  confidence: number;
}

export interface SearchResponse {
  query: string;
  mode: string;
  reranked: boolean;
  total_results: number;
  latency_ms: number;
  retrieval_latency_ms: number;
  results: BookResult[];
  query_understanding?: QueryUnderstanding;
}

export interface AskResponse {
  answer: string;
  sources: BookResult[];
  latency_ms: number;
  generation_latency_ms: number;
  retrieval_latency_ms: number;
  citations_valid: boolean;
}

export async function searchBooks(
  query: string,
  options: {
    mode?: "hybrid" | "vector" | "keyword";
    top_k?: number;
    understand?: boolean;
  } = {}
): Promise<SearchResponse> {
  const params = new URLSearchParams({ q: query });
  if (options.mode) params.set("mode", options.mode);
  if (options.top_k) params.set("top_k", String(options.top_k));
  if (options.understand !== undefined)
    params.set("understand", String(options.understand));

  const res = await fetch(`${API_BASE}/search?${params}`);
  if (!res.ok) throw new Error(`Search failed: ${res.status}`);
  return res.json();
}

export async function askBooks(question: string): Promise<AskResponse> {
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: 8 }),
  });
  if (!res.ok) throw new Error(`Ask failed: ${res.status}`);
  return res.json();
}

export async function browseCatalog(
  top_k: number = 20
): Promise<SearchResponse> {
  // Use a broad query to get a diverse catalog sample
  const res = await fetch(
    `${API_BASE}/search?q=*&mode=keyword&top_k=${top_k}&understand=false`
  );
  if (!res.ok) throw new Error(`Browse failed: ${res.status}`);
  return res.json();
}
