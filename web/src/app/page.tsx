"use client";

import { useState, useEffect, useCallback } from "react";
import { SearchBar } from "@/components/search-bar";
import { BookCard } from "@/components/book-card";
import { ModeToggle } from "@/components/mode-toggle";
import { CompareView } from "@/components/compare-view";
import { AskView } from "@/components/ask-view";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  searchBooks,
  BookResult,
  QueryUnderstanding,
} from "@/lib/api";
import { BookOpen, GitCompare, Zap, Search, MessageCircle } from "lucide-react";

const SAMPLE_QUERIES = [
  "fantasy adventure",
  "history of science",
  "romance novels",
  "mystery detective",
  "philosophy",
  "cooking mediterranean",
];

export default function Home() {
  const [results, setResults] = useState<BookResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<"hybrid" | "vector" | "keyword">("hybrid");
  const [latencyMs, setLatencyMs] = useState<number | undefined>();
  const [totalResults, setTotalResults] = useState(0);
  const [queryUnderstanding, setQueryUnderstanding] =
    useState<QueryUnderstanding | null>(null);
  const [spellCorrection, setSpellCorrection] = useState(true);
  const [rerank, setRerank] = useState(false);
  const [compareMode, setCompareMode] = useState(false);
  const [currentQuery, setCurrentQuery] = useState("");
  const [catalogLoaded, setCatalogLoaded] = useState(false);
  const [activeTab, setActiveTab] = useState("search");

  useEffect(() => {
    loadCatalogSample();
  }, []);

  const loadCatalogSample = async () => {
    try {
      setLoading(true);
      const randomQuery =
        SAMPLE_QUERIES[Math.floor(Math.random() * SAMPLE_QUERIES.length)];
      const res = await searchBooks(randomQuery, {
        mode: "hybrid",
        top_k: 12,
        understand: false,
      });
      setResults(res.results);
      setCatalogLoaded(true);
    } catch {
      setCatalogLoaded(false);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = useCallback(
    async (query: string) => {
      if (compareMode) {
        setCurrentQuery(query);
        return;
      }
      setLoading(true);
      setCurrentQuery(query);
      try {
        const res = await searchBooks(query, {
          mode,
          top_k: 10,
          understand: spellCorrection,
          rerank,
        });
        setResults(res.results);
        setLatencyMs(res.latency_ms);
        setTotalResults(res.total_results);
        setQueryUnderstanding(res.query_understanding || null);
      } catch {
        setResults([]);
        setQueryUnderstanding(null);
      } finally {
        setLoading(false);
      }
    },
    [mode, spellCorrection, compareMode, rerank]
  );

  useEffect(() => {
    if (currentQuery && !compareMode) {
      handleSearch(currentQuery);
    }
  }, [mode]);

  return (
    <main className="flex-1 flex flex-col">
      {/* Header */}
      <header className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <BookOpen className="h-5 w-5 text-primary" />
              <h1 className="text-lg font-semibold">BookSearch</h1>
              <Badge variant="secondary" className="text-[10px]">
                AI-Powered
              </Badge>
            </div>

            <div className="flex items-center gap-4">
              {/* Tab switcher */}
              <div className="flex gap-1 p-1 bg-muted rounded-lg">
                <button
                  onClick={() => setActiveTab("search")}
                  className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all flex items-center gap-1.5 ${
                    activeTab === "search"
                      ? "bg-background text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <Search className="h-3 w-3" />
                  Search
                </button>
                <button
                  onClick={() => setActiveTab("ask")}
                  className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all flex items-center gap-1.5 ${
                    activeTab === "ask"
                      ? "bg-background text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <MessageCircle className="h-3 w-3" />
                  Ask
                </button>
              </div>

              {activeTab === "search" && (
                <>
                  <div className="flex items-center gap-2">
                    <Switch
                      id="spell"
                      checked={spellCorrection}
                      onCheckedChange={setSpellCorrection}
                    />
                    <Label htmlFor="spell" className="text-xs text-muted-foreground">
                      Spell fix
                    </Label>
                  </div>

                  <div className="flex items-center gap-2">
                    <Switch
                      id="rerank"
                      checked={rerank}
                      onCheckedChange={setRerank}
                    />
                    <Label htmlFor="rerank" className="text-xs text-muted-foreground">
                      <Zap className="h-3 w-3 inline mr-1" />
                      Rerank
                    </Label>
                  </div>

                  <div className="flex items-center gap-2">
                    <Switch
                      id="compare"
                      checked={compareMode}
                      onCheckedChange={setCompareMode}
                    />
                    <Label htmlFor="compare" className="text-xs text-muted-foreground">
                      <GitCompare className="h-3 w-3 inline mr-1" />
                      Compare
                    </Label>
                  </div>
                </>
              )}
            </div>
          </div>

          {activeTab === "search" && (
            <>
              <SearchBar
                onSearch={handleSearch}
                loading={loading}
                queryUnderstanding={queryUnderstanding}
              />

              {!compareMode && (
                <div className="flex items-center justify-between mt-3">
                  <ModeToggle
                    selected={mode}
                    onChange={setMode}
                    latencyMs={latencyMs}
                  />
                  {totalResults > 0 && (
                    <span className="text-xs text-muted-foreground">
                      {totalResults.toLocaleString()} results
                    </span>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </header>

      {/* Content */}
      <div className="flex-1 max-w-5xl mx-auto w-full px-4 py-6">
        {activeTab === "ask" ? (
          <AskView />
        ) : compareMode && currentQuery ? (
          <CompareView query={currentQuery} spellCorrection={spellCorrection} />
        ) : (
          <>
            {!currentQuery && catalogLoaded && (
              <div className="mb-4">
                <p className="text-sm text-muted-foreground mb-3">
                  <Zap className="h-3.5 w-3.5 inline mr-1" />
                  Explore the catalog — 26K+ books with hybrid semantic search
                </p>
              </div>
            )}

            <div className="grid gap-3">
              {results.map((book) => (
                <BookCard key={book.id} book={book} />
              ))}
            </div>

            {!loading && results.length === 0 && currentQuery && (
              <div className="text-center py-12 text-muted-foreground">
                <p>No results found for &ldquo;{currentQuery}&rdquo;</p>
                <p className="text-sm mt-1">
                  Try a different query or switch search mode
                </p>
              </div>
            )}

            {!catalogLoaded && !loading && (
              <div className="text-center py-12 text-muted-foreground">
                <BookOpen className="h-8 w-8 mx-auto mb-3 opacity-50" />
                <p className="font-medium">API not available</p>
                <p className="text-sm mt-1">
                  Start the backend with{" "}
                  <code className="bg-muted px-1.5 py-0.5 rounded text-xs">
                    python -m uvicorn src.api.main:app
                  </code>
                </p>
              </div>
            )}
          </>
        )}
      </div>

      {/* Footer */}
      <footer className="border-t py-3 text-center text-xs text-muted-foreground">
        Hybrid Search (TF-IDF + Vector + RRF) · nomic-embed-text-v1.5 · Qdrant
        · Cross-Encoder Reranker
      </footer>
    </main>
  );
}
