"""Query understanding pipeline — spell correction → intent classification.

Orchestrates the full query processing before retrieval:
1. Spell correction (SymSpell, <1ms)
2. Intent classification (rule-based, <1ms)
3. Returns QueryAnalysis with optimal search parameters

Usage:
    from src.query.pipeline import QueryPipeline
    
    qp = QueryPipeline()
    analysis = qp.process("books liek project hail mary")
    # → corrected="books like project hail mary", intent=similar_to
"""

from dataclasses import replace

from src.query.spell import SpellCorrector
from src.query.intent import classify_intent, QueryAnalysis


class QueryPipeline:
    """Full query understanding pipeline."""

    def __init__(self):
        self.spell = SpellCorrector()
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self.spell.load()
            self._loaded = True

    def process(self, query: str) -> QueryAnalysis:
        """Process a raw query through spell correction and intent classification."""
        self._ensure_loaded()

        # Step 1: Spell correction
        corrected, was_corrected = self.spell.correct(query)

        # Step 2: Intent classification (on corrected query)
        analysis = classify_intent(corrected)

        # Update with correction info
        analysis = replace(
            analysis,
            original=query,
            corrected=corrected,
            was_corrected=was_corrected,
        )

        return analysis


if __name__ == "__main__":
    import time

    qp = QueryPipeline()

    test_queries = [
        # Typos
        "romanse set in scotlnd",
        "books liek project hail mary",
        "jule garwood novels",
        # Intent: similar_to
        "books similar to The Great Gatsby",
        "if I liked Dune what should I read",
        # Intent: author
        "books by Stephen King",
        "Margaret Atwood novels",
        # Intent: filtered
        "thriller published after 2015",
        "classic literature from the 1800s",
        # Intent: concept/semantic
        "books about loneliness and isolation",
        "stories exploring the meaning of home",
        # Intent: keyword (ISBN)
        "978-0-13-468599-1",
        # General
        "world war 2 memoir",
        "cooking italian food",
    ]

    print(f"{'Query':<45} {'Corrected':<45} {'Intent':<12} {'Mode':<8}")
    print("-" * 115)

    for q in test_queries:
        t0 = time.time()
        result = qp.process(q)
        elapsed = (time.time() - t0) * 1000

        corrected_display = result.corrected if result.was_corrected else "—"
        print(
            f"{q:<45} "
            f"{corrected_display:<45} "
            f"{result.intent.value:<12} "
            f"{result.search_mode:<8}"
        )
