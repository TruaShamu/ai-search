"""Spell correction using SymSpell with a domain-specific dictionary.

Builds dictionary from book titles, authors, and subjects in the index.
SymSpell uses symmetric delete algorithm — O(1) lookups after dictionary build.
"""

import os
from pathlib import Path

from symspellpy import SymSpell, Verbosity
from dotenv import load_dotenv

load_dotenv()

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

    Extracts unique terms from titles, authors, and subjects.
    Each term gets frequency=1000 (high enough to prefer over generic English).
    """
    import requests

    # Pull terms from Azure AI Search index
    endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
    api_key = os.environ["AZURE_SEARCH_ADMIN_KEY"]
    index = os.environ.get("AZURE_SEARCH_INDEX", "books-v1")

    headers = {"Content-Type": "application/json", "api-key": api_key}
    url = f"{endpoint}/indexes/{index}/docs/search?api-version=2024-07-01"

    # Fetch a sample of books to build vocab
    all_terms = set()
    batch_size = 1000

    for skip in range(0, 13500, batch_size):
        body = {
            "search": "*",
            "top": batch_size,
            "skip": skip,
            "select": "title,authors,subjects",
        }
        resp = requests.post(url, headers=headers, json=body)
        if resp.status_code != 200:
            break

        for doc in resp.json().get("value", []):
            # Extract terms from title
            title = doc.get("title", "")
            for word in title.lower().split():
                if len(word) >= 3 and word.isalpha():
                    all_terms.add(word)

            # Extract terms from authors
            authors = doc.get("authors", "")
            for word in authors.lower().split():
                if len(word) >= 3 and word.isalpha():
                    all_terms.add(word)

            # Extract terms from subjects
            for subject in doc.get("subjects", []):
                for word in subject.lower().split():
                    word = word.strip(",.;:()")
                    if len(word) >= 3 and word.isalpha():
                        all_terms.add(word)

        print(f"  Processed {skip + batch_size} docs, {len(all_terms)} unique terms")

    # Write dictionary file (term<space>frequency per line)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for term in sorted(all_terms):
            f.write(f"{term} 1000\n")

    print(f"\nCorpus dictionary: {len(all_terms)} terms -> {output_path}")
    return len(all_terms)


if __name__ == "__main__":
    print("Building corpus dictionary from Azure AI Search index...")
    build_corpus_dictionary()
