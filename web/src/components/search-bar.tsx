"use client";

import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Search, Sparkles } from "lucide-react";

interface SearchBarProps {
  onSearch: (query: string) => void;
  loading?: boolean;
  queryUnderstanding?: {
    original: string;
    corrected: string;
    was_corrected: boolean;
    intent: string;
  } | null;
}

export function SearchBar({ onSearch, loading, queryUnderstanding }: SearchBarProps) {
  const [query, setQuery] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) onSearch(query.trim());
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <form onSubmit={handleSubmit} className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search books... try &quot;romance set in Scotland&quot; or &quot;books about loneliness&quot;"
          className="pl-10 pr-4 h-12 text-base"
          disabled={loading}
        />
        {loading && (
          <div className="absolute right-3 top-1/2 -translate-y-1/2">
            <div className="h-4 w-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
          </div>
        )}
      </form>

      {/* Spell correction indicator */}
      {queryUnderstanding?.was_corrected && (
        <div className="mt-2 flex items-center gap-2 text-sm text-muted-foreground">
          <Sparkles className="h-3.5 w-3.5 text-amber-500" />
          <span>
            Showing results for{" "}
            <span className="font-medium text-foreground">
              {queryUnderstanding.corrected}
            </span>
          </span>
          <Badge variant="outline" className="text-[10px]">
            {queryUnderstanding.intent}
          </Badge>
        </div>
      )}

      {/* Intent badge (when no correction) */}
      {queryUnderstanding && !queryUnderstanding.was_corrected && (
        <div className="mt-2 flex items-center gap-2">
          <Badge variant="secondary" className="text-[10px]">
            {queryUnderstanding.intent}
          </Badge>
        </div>
      )}
    </div>
  );
}
