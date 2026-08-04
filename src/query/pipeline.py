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
