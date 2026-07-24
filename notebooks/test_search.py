"""Quick search test — validate embedding quality with sample queries."""
from pathlib import Path
from src.search.engine import BookSearchEngine

engine = BookSearchEngine(Path("data/index"))

queries = [
    "books about loneliness in space",
    "history of computing and the internet",
    "romance set in Scotland",
    "philosophy of meaning and suffering",
    "children's adventure fantasy",
    "world war 2 memoir",
    "african american literature",
    "cooking and food culture",
]

for q in queries:
    print(engine.search_formatted(q, top_k=5))
    print("=" * 70)
    print()
