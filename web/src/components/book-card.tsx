"use client";

import { useState } from "react";
import { BookResult } from "@/lib/api";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { BookImage } from "lucide-react";

export function BookCard({ book, compact }: { book: BookResult; compact?: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const [coverFailed, setCoverFailed] = useState(false);
  const coverUrl =
    book.cover_url ||
    (book.work_id
      ? `https://covers.openlibrary.org/b/olid/${book.work_id}-M.jpg?default=false`
      : null);

  const hasLongDesc = book.description && book.description.length > 100;

  return (
    <Card
      className={`hover:ring-2 hover:ring-primary/30 transition-all ${
        !compact && hasLongDesc ? "cursor-pointer" : ""
      }`}
      onClick={() => {
        if (!compact && hasLongDesc) setExpanded(!expanded);
      }}
    >
      <CardContent className={`flex gap-3 ${compact ? "p-2.5" : "p-4"}`}>
        {/* Cover thumbnail */}
        {!compact && (
          <div className="flex-shrink-0 w-16 h-24 bg-muted rounded overflow-hidden">
            {coverUrl && !coverFailed ? (
              <img
                src={coverUrl}
                alt={`${book.title} cover`}
                className="w-full h-full object-cover"
                onError={() => setCoverFailed(true)}
              />
            ) : (
              <div
                className="flex h-full w-full items-center justify-center border border-border/70 bg-muted/70 text-muted-foreground"
                aria-label={`No cover available for ${book.title}`}
                role="img"
              >
                <BookImage className="h-8 w-8" strokeWidth={1.5} aria-hidden="true" />
              </div>
            )}
          </div>
        )}

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <h3 className={`font-semibold leading-tight ${compact ? "text-xs truncate" : "text-sm"}`}>
              {book.title}
            </h3>
            {/* Score */}
            {book.score > 0 && (
              <span className="flex-shrink-0 text-[10px] text-muted-foreground font-mono">
                {book.score.toFixed(3)}
              </span>
            )}
          </div>
          <p className="text-xs text-muted-foreground mt-0.5">
            {book.authors || "Unknown author"}
            {book.year && ` · ${book.year}`}
          </p>

          {/* Description: clamped by default, full on expand */}
          {!compact && book.description && (
            <p
              className={`text-xs text-muted-foreground mt-1.5 ${
                expanded ? "" : "line-clamp-2"
              }`}
            >
              {book.description}
            </p>
          )}

          {/* Expand hint */}
          {!compact && hasLongDesc && !expanded && (
            <span className="text-[10px] text-primary/70 mt-0.5 inline-block">
              click to expand ↓
            </span>
          )}

          {/* Subject badges */}
          {book.subjects && book.subjects.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-1.5">
              {book.subjects.slice(0, compact ? 2 : 3).map((s) => (
                <Badge key={s} variant="secondary" className="text-[10px] px-1.5 py-0">
                  {s}
                </Badge>
              ))}
              {book.subjects.length > (compact ? 2 : 3) && (
                <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                  +{book.subjects.length - (compact ? 2 : 3)}
                </Badge>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
