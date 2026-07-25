"use client";

import { BookResult } from "@/lib/api";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export function BookCard({ book }: { book: BookResult }) {
  const coverUrl =
    book.cover_url || `https://covers.openlibrary.org/b/olid/${book.work_id}-M.jpg`;

  return (
    <Card className="hover:ring-2 hover:ring-primary/30 transition-all">
      <CardContent className="flex gap-4 p-4">
        {/* Cover thumbnail */}
        <div className="flex-shrink-0 w-16 h-24 bg-muted rounded overflow-hidden">
          <img
            src={coverUrl}
            alt={book.title}
            className="w-full h-full object-cover"
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-sm leading-tight truncate">
            {book.title}
          </h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            {book.authors || "Unknown author"}
            {book.year && ` · ${book.year}`}
          </p>

          {book.description && (
            <p className="text-xs text-muted-foreground mt-1.5 line-clamp-2">
              {book.description}
            </p>
          )}

          {/* Subject badges */}
          {book.subjects && book.subjects.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {book.subjects.slice(0, 3).map((s) => (
                <Badge key={s} variant="secondary" className="text-[10px] px-1.5 py-0">
                  {s}
                </Badge>
              ))}
              {book.subjects.length > 3 && (
                <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                  +{book.subjects.length - 3}
                </Badge>
              )}
            </div>
          )}
        </div>

        {/* Score */}
        {book.score > 0 && (
          <div className="flex-shrink-0 text-right">
            <span className="text-[10px] text-muted-foreground font-mono">
              {book.score.toFixed(3)}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
