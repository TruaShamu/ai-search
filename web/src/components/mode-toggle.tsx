"use client";

import { Badge } from "@/components/ui/badge";

interface SearchMode {
  value: "hybrid" | "vector" | "keyword";
  label: string;
  description: string;
}

const MODES: SearchMode[] = [
  { value: "hybrid", label: "Hybrid", description: "TF-IDF + Vector + RRF" },
  { value: "vector", label: "Semantic", description: "Vector similarity only" },
  { value: "keyword", label: "Keyword", description: "TF-IDF text matching" },
];

interface ModeToggleProps {
  selected: "hybrid" | "vector" | "keyword";
  onChange: (mode: "hybrid" | "vector" | "keyword") => void;
  latencyMs?: number;
}

export function ModeToggle({ selected, onChange, latencyMs }: ModeToggleProps) {
  return (
    <div className="flex items-center gap-3">
      <div className="flex gap-1 p-1 bg-muted rounded-lg">
        {MODES.map((mode) => (
          <button
            key={mode.value}
            onClick={() => onChange(mode.value)}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
              selected === mode.value
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
            title={mode.description}
          >
            {mode.label}
          </button>
        ))}
      </div>

      {latencyMs !== undefined && (
        <Badge variant="outline" className="text-[10px] font-mono">
          {latencyMs.toFixed(0)}ms
        </Badge>
      )}
    </div>
  );
}
