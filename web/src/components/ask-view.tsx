"use client";

import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { BookCard } from "@/components/book-card";
import { askBooks, AskResponse } from "@/lib/api";
import { MessageCircle, Send, BookOpen } from "lucide-react";

export function AskView() {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleAsk = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError("");
    try {
      const res = await askBooks(question.trim());
      setResponse(res);
    } catch (err) {
      setError("Failed to get answer. Is the API running?");
      setResponse(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="text-center mb-6">
        <MessageCircle className="h-6 w-6 mx-auto mb-2 text-primary" />
        <p className="text-sm text-muted-foreground">
          Ask questions about books — answers are grounded in search results with citations
        </p>
      </div>

      {/* Question input */}
      <form onSubmit={handleAsk} className="flex gap-2 max-w-2xl mx-auto">
        <Input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask anything... &quot;What are good books about stoicism?&quot;"
          className="h-11"
          disabled={loading}
        />
        <Button type="submit" disabled={loading || !question.trim()} size="default">
          {loading ? (
            <div className="h-4 w-4 border-2 border-primary-foreground border-t-transparent rounded-full animate-spin" />
          ) : (
            <Send className="h-4 w-4" />
          )}
        </Button>
      </form>

      {/* Error */}
      {error && (
        <p className="text-sm text-destructive text-center">{error}</p>
      )}

      {/* Answer */}
      {response && (
        <div className="max-w-2xl mx-auto space-y-4">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-start gap-3">
                <BookOpen className="h-4 w-4 mt-1 text-primary flex-shrink-0" />
                <div className="flex-1">
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">
                    {response.answer}
                  </p>
                  <div className="flex items-center gap-2 mt-3 pt-3 border-t">
                    <Badge variant="outline" className="text-[10px] font-mono">
                      {response.retrieval_latency_ms?.toFixed(0)}ms retrieval
                    </Badge>
                    <Badge variant="outline" className="text-[10px] font-mono">
                      {response.generation_latency_ms?.toFixed(0)}ms generation
                    </Badge>
                    {response.citations_valid && (
                      <Badge variant="secondary" className="text-[10px]">
                        ✓ citations verified
                      </Badge>
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Sources */}
          {response.sources && response.sources.length > 0 && (
            <div>
              <p className="text-xs text-muted-foreground mb-2 font-medium">
                Sources ({response.sources.length} books retrieved)
              </p>
              <div className="grid gap-2">
                {response.sources.map((book, i) => (
                  <BookCard key={book.id || i} book={book} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
