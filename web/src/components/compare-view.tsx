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
  const [leftResult, setLeftResult] = useState<SearchResponse | null>(null);
  const [rightResult, setRightResult] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!query) return;

    const fetchBoth = async () => {
      setLoading(true);
      try {
        const [keyword, vector] = await Promise.all([
          searchBooks(query, { mode: "keyword", top_k: 8, understand: spellCorrection }),
          searchBooks(query, { mode: "vector", top_k: 8, understand: spellCorrection }),
        ]);
        setLeftResult(keyword);
        setRightResult(vector);
      } catch {
        setLeftResult(null);
        setRightResult(null);
      } finally {
        setLoading(false);
      }
    };

    fetchBoth();
  }, [query, spellCorrection]);

  if (loading) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        <div className="h-5 w-5 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-2" />
        Comparing search modes...
      </div>
    );
  }

  if (!leftResult || !rightResult) return null;

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground text-center">
        Side-by-side: how different retrieval strategies rank results for the same query
      </p>

      <div className="grid grid-cols-2 gap-6">
        {/* Keyword (BM25) column */}
        <div>
          <div className="flex items-center gap-2 mb-3 pb-2 border-b">
            <Badge variant="outline">Keyword (BM25)</Badge>
            <span className="text-[10px] text-muted-foreground font-mono">
              {leftResult.latency_ms.toFixed(0)}ms
            </span>
          </div>
          <div className="grid gap-2">
            {leftResult.results.map((book) => (
              <BookCard key={book.id} book={book} />
            ))}
          </div>
        </div>

        {/* Vector (Semantic) column */}
        <div>
          <div className="flex items-center gap-2 mb-3 pb-2 border-b">
            <Badge variant="outline">Semantic (Vector)</Badge>
            <span className="text-[10px] text-muted-foreground font-mono">
              {rightResult.latency_ms.toFixed(0)}ms
            </span>
          </div>
          <div className="grid gap-2">
            {rightResult.results.map((book) => (
              <BookCard key={book.id} book={book} />
            ))}
          </div>
        </div>
      </div>

      {/* Overlap analysis */}
      <div className="mt-4 p-3 bg-muted/50 rounded-lg text-center">
        <OverlapBadge left={leftResult} right={rightResult} />
      </div>
    </div>
  );
}

function OverlapBadge({
  left,
  right,
}: {
  left: SearchResponse;
  right: SearchResponse;
}) {
  const leftIds = new Set(left.results.map((r) => r.id));
  const rightIds = new Set(right.results.map((r) => r.id));
  const overlap = [...leftIds].filter((id) => rightIds.has(id)).length;
  const total = Math.max(leftIds.size, rightIds.size);
  const pct = total > 0 ? Math.round((overlap / total) * 100) : 0;

  return (
    <span className="text-xs text-muted-foreground">
      <span className="font-medium text-foreground">{overlap}</span> shared results
      ({pct}% overlap) — {total - overlap} unique per mode
    </span>
  );
}
