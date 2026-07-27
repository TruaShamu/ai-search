"""Spell correction using SymSpell with a domain-specific dictionary.

Builds dictionary from book titles, authors, and subjects in the corpus.
SymSpell uses symmetric delete algorithm — O(1) lookups after dictionary build.
"""

from pathlib import Path

from symspellpy import SymSpell, Verbosity

# Default frequency dictionary (English)
FREQ_DICT_PATH = Path(__file__).parent / "frequency_dictionary.txt"
CORPUS_DICT_PATH = Path("data/processed/corpus_dictionary.txt")


class SpellCorrector:
    """SymSpell-based spell correction with book-domain vocabulary."""

    def __init__(self, max_edit_distance: int = 2):
        self.sym = SymSpell(max_dictionary_edit_distance=max_edit_distance)
        self.max_edit_distance = max_edit_distance
        self._loaded = False

    def load(self):
        """Load dictionaries — English baseline + corpus-specific terms."""
        # Load the built-in English frequency dictionary from symspellpy
        import symspellpy
        pkg_path = Path(symspellpy.__file__).parent
        dict_path = pkg_path / "frequency_dictionary_en_82_765.txt"

        if dict_path.exists():
            self.sym.load_dictionary(str(dict_path), 0, 1, encoding="utf-8")

        # Load corpus dictionary if available (titles, authors, subjects)
        if CORPUS_DICT_PATH.exists():
            self.sym.load_dictionary(str(CORPUS_DICT_PATH), 0, 1, encoding="utf-8")

        self._loaded = True

    def correct(self, query: str) -> tuple[str, bool]:
        """
        Correct spelling in a query.

        Returns:
            (corrected_query, was_corrected)
        """
        if not self._loaded:
            self.load()

        words = query.split()
        corrected_words = []
        any_corrected = False

        for word in words:
            # Skip short words, numbers, and likely proper nouns (capitalized mid-query)
            if len(word) <= 2 or word.isdigit():
                corrected_words.append(word)
                continue

            suggestions = self.sym.lookup(
                word.lower(),
                Verbosity.CLOSEST,
                max_edit_distance=self.max_edit_distance,
            )

            if suggestions and suggestions[0].distance > 0:
                corrected_words.append(suggestions[0].term)
                any_corrected = True
            else:
                corrected_words.append(word)

        return " ".join(corrected_words), any_corrected

    def suggest(self, query: str, max_suggestions: int = 3) -> list[str]:
        """Get alternative query suggestions (did-you-mean)."""
        if not self._loaded:
            self.load()

        suggestions = self.sym.lookup_compound(
            query,
            max_edit_distance=self.max_edit_distance,
        )

        return [s.term for s in suggestions[:max_suggestions] if s.term != query.lower()]


def build_corpus_dictionary(output_path: Path = CORPUS_DICT_PATH):
    """Build a domain dictionary from our book corpus for SymSpell.

    Reads data/processed/books_augmented.jsonl directly — no network call needed.
    Only includes the indexed books (those with a non-empty "description" field),
    which matches the 26,519 books actually searchable in Qdrant.

    Extracts unique terms from titles, authors, and subjects.
    Each term gets frequency=1000 (high enough to prefer over generic English).
    """
    import json

    jsonl_path = Path("data/processed/books_augmented.jsonl")
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Missing corpus file: {jsonl_path}. "
            "Download or regenerate books_augmented.jsonl first."
        )

    all_terms: set[str] = set()
    indexed_count = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            # Only include indexed books (non-empty description)
            if not doc.get("description"):
                continue
            indexed_count += 1

            # Extract terms from title
            title = doc.get("title") or ""
            for word in title.lower().split():
                if len(word) >= 3 and word.isalpha():
                    all_terms.add(word)

            # Extract terms from authors
            authors = doc.get("authors") or []
            if isinstance(authors, str):
                authors = [authors]
            for author in authors:
                for word in author.lower().split():
                    if len(word) >= 3 and word.isalpha():
                        all_terms.add(word)

            # Extract terms from subjects
            for subject in doc.get("subjects") or []:
                for word in subject.lower().split():
                    word = word.strip(",.;:()")
                    if len(word) >= 3 and word.isalpha():
                        all_terms.add(word)

    # Write dictionary file (term<space>frequency per line)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for term in sorted(all_terms):
            f.write(f"{term} 1000\n")

    print(f"\nCorpus dictionary: {len(all_terms)} terms from {indexed_count} indexed books -> {output_path}")
    return len(all_terms)


if __name__ == "__main__":
    print("Building corpus dictionary from books_augmented.jsonl...")
    build_corpus_dictionary()
