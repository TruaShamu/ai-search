"use client";

import { useState, useEffect } from "react";
import { BookCard } from "@/components/book-card";
import { Badge } from "@/components/ui/badge";
import { searchBooks, SearchResponse } from "@/lib/api";

interface CompareViewProps {
  query: string;
  spellCorrection: boolean;
}

export function CompareView({ query, spellCorrection }: CompareViewProps) {
  const [keywordResult, setKeywordResult] = useState<SearchResponse | null>(null);
  const [vectorResult, setVectorResult] = useState<SearchResponse | null>(null);
  const [hybridResult, setHybridResult] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!query) return;

    const fetchAll = async () => {
      setLoading(true);
      try {
        const [keyword, vector, hybrid] = await Promise.all([
          searchBooks(query, { mode: "keyword", top_k: 8, understand: spellCorrection }),
          searchBooks(query, { mode: "vector", top_k: 8, understand: spellCorrection }),
          searchBooks(query, { mode: "hybrid", top_k: 8, understand: spellCorrection }),
        ]);
        setKeywordResult(keyword);
        setVectorResult(vector);
        setHybridResult(hybrid);
      } catch {
        setKeywordResult(null);
        setVectorResult(null);
        setHybridResult(null);
      } finally {
        setLoading(false);
      }
    };

    fetchAll();
  }, [query, spellCorrection]);

  if (loading) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        <div className="h-5 w-5 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-2" />
        Comparing search modes...
      </div>
    );
  }

  if (!keywordResult || !vectorResult || !hybridResult) return null;

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground text-center">
        Side-by-side: how BM25, Vector, and Hybrid (RRF) rank results for the same query
      </p>

      <div className="grid grid-cols-3 gap-4">
        {/* Keyword (BM25) column */}
        <div>
          <div className="flex items-center gap-2 mb-3 pb-2 border-b">
            <Badge variant="outline">BM25</Badge>
            <span className="text-[10px] text-muted-foreground font-mono">
              {keywordResult.latency_ms.toFixed(0)}ms
            </span>
          </div>
          <div className="grid gap-2">
            {keywordResult.results.map((book) => (
              <BookCard key={book.id} book={book} compact />
            ))}
          </div>
        </div>

        {/* Hybrid (RRF) column — center, highlighted */}
        <div>
          <div className="flex items-center gap-2 mb-3 pb-2 border-b border-primary/40">
            <Badge>Hybrid (RRF)</Badge>
            <span className="text-[10px] text-muted-foreground font-mono">
              {hybridResult.latency_ms.toFixed(0)}ms
            </span>
          </div>
          <div className="grid gap-2">
            {hybridResult.results.map((book) => (
              <BookCard key={book.id} book={book} compact />
            ))}
          </div>
        </div>

        {/* Vector (Semantic) column */}
        <div>
          <div className="flex items-center gap-2 mb-3 pb-2 border-b">
            <Badge variant="outline">Vector</Badge>
            <span className="text-[10px] text-muted-foreground font-mono">
              {vectorResult.latency_ms.toFixed(0)}ms
            </span>
          </div>
          <div className="grid gap-2">
            {vectorResult.results.map((book) => (
              <BookCard key={book.id} book={book} compact />
            ))}
          </div>
        </div>
      </div>

      {/* Overlap analysis */}
      <div className="mt-4 p-3 bg-muted/50 rounded-lg text-center space-y-1">
        <OverlapBadge label="BM25 ∩ Vector" left={keywordResult} right={vectorResult} />
        <OverlapBadge label="BM25 ∩ Hybrid" left={keywordResult} right={hybridResult} />
        <OverlapBadge label="Vector ∩ Hybrid" left={vectorResult} right={hybridResult} />
      </div>
    </div>
  );
}

function OverlapBadge({
  label,
  left,
  right,
}: {
  label: string;
  left: SearchResponse;
  right: SearchResponse;
}) {
  const leftIds = new Set(left.results.map((r) => r.id));
  const rightIds = new Set(right.results.map((r) => r.id));
  const overlap = [...leftIds].filter((id) => rightIds.has(id)).length;
  const total = Math.max(leftIds.size, rightIds.size);
  const pct = total > 0 ? Math.round((overlap / total) * 100) : 0;

  return (
    <span className="text-xs text-muted-foreground block">
      <span className="font-mono">{label}:</span>{" "}
      <span className="font-medium text-foreground">{overlap}</span> shared
      ({pct}% overlap)
    </span>
  );
}
