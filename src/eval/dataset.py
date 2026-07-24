"""Evaluation dataset for book search.

Contains query-relevance pairs for evaluating retrieval quality.
Each entry has:
  - query: the search query
  - relevant: list of {work_id, relevance} where relevance is 0/1/2
  - source: "synthetic" or "manual"
  - category: query type (topic, author, genre, concept, etc.)
"""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict

EVAL_DATA_PATH = Path("data/eval/queries.json")


@dataclass
class RelevanceJudgment:
    work_id: str
    relevance: int  # 0=not relevant, 1=partial, 2=highly relevant
    title: str = ""  # for readability


@dataclass
class EvalQuery:
    query: str
    relevant: list[RelevanceJudgment] = field(default_factory=list)
    source: str = "synthetic"  # synthetic | manual
    category: str = "topic"  # topic | genre | concept | author | era


# --- Synthetic eval dataset ---
# Generated from known books in our index. For each query, we know which books
# should appear because we derived the query from their metadata.

EVAL_QUERIES = [
    # === GENRE queries ===
    EvalQuery(
        query="romance set in Scotland",
        category="genre",
        relevant=[
            RelevanceJudgment("OL18178460W", 2, "Seducing the Highlander"),
            RelevanceJudgment("OL3277302W", 2, "Border Bride"),
            RelevanceJudgment("OL26945W", 2, "The Bride"),
        ],
    ),
    EvalQuery(
        query="mystery detective noir",
        category="genre",
        relevant=[
            RelevanceJudgment("OL17350938W", 2, "The Art of Detection"),
        ],
    ),
    EvalQuery(
        query="fantasy adventure for children",
        category="genre",
        relevant=[
            RelevanceJudgment("OL20923796W", 2, "Jungle Adventures"),
        ],
    ),
    EvalQuery(
        query="historical fiction about ancient Rome",
        category="genre",
        relevant=[],  # to be judged manually
    ),
    EvalQuery(
        query="horror supernatural ghost story",
        category="genre",
        relevant=[
            RelevanceJudgment("OL20660197W", 1, "The Haunted"),
        ],
    ),

    # === TOPIC queries ===
    EvalQuery(
        query="history of computing and the internet",
        category="topic",
        relevant=[
            RelevanceJudgment("OL1965741W", 2, "The Internet in Everyday Life"),
            RelevanceJudgment("OL277729W", 1, "The Road Ahead"),
        ],
    ),
    EvalQuery(
        query="cooking and food culture",
        category="topic",
        relevant=[
            RelevanceJudgment("OL32136968W", 2, "Celebration food"),
        ],
    ),
    EvalQuery(
        query="world war 2 memoir",
        category="topic",
        relevant=[
            RelevanceJudgment("OL24227892W", 2, "The war-time diary of Gertrude Elizabeth Bathurst"),
            RelevanceJudgment("OL5703691W", 2, "Serenade to the Big Bird"),
            RelevanceJudgment("OL5857618W", 1, "We Who Are Alive and Remain"),
        ],
    ),
    EvalQuery(
        query="african american literature and identity",
        category="topic",
        relevant=[
            RelevanceJudgment("OL3949189W", 2, "Cane River"),
        ],
    ),
    EvalQuery(
        query="climate change and environment",
        category="topic",
        relevant=[],
    ),
    EvalQuery(
        query="artificial intelligence and machine learning",
        category="topic",
        relevant=[],
    ),
    EvalQuery(
        query="psychology of habits and behavior",
        category="topic",
        relevant=[],
    ),
    EvalQuery(
        query="space exploration and astronomy",
        category="topic",
        relevant=[],
    ),
    EvalQuery(
        query="philosophy of meaning and suffering",
        category="topic",
        relevant=[],
    ),

    # === CONCEPT queries (abstract, harder for keyword search) ===
    EvalQuery(
        query="books about loneliness and isolation",
        category="concept",
        relevant=[],
    ),
    EvalQuery(
        query="stories about redemption and second chances",
        category="concept",
        relevant=[],
    ),
    EvalQuery(
        query="overcoming adversity and personal growth",
        category="concept",
        relevant=[],
    ),
    EvalQuery(
        query="the meaning of home and belonging",
        category="concept",
        relevant=[],
    ),
    EvalQuery(
        query="forbidden love across social classes",
        category="concept",
        relevant=[],
    ),

    # === SPECIFIC queries (should reward exact matching) ===
    EvalQuery(
        query="Bill Gates book about technology future",
        category="specific",
        relevant=[
            RelevanceJudgment("OL277729W", 2, "The Road Ahead"),
        ],
    ),
    EvalQuery(
        query="Julie Garwood scottish romance novel",
        category="specific",
        relevant=[
            RelevanceJudgment("OL26945W", 2, "The Bride"),
        ],
    ),
    EvalQuery(
        query="Maya Angelou autobiography",
        category="specific",
        relevant=[],
    ),
    EvalQuery(
        query="italian mediterranean cookbook recipes",
        category="specific",
        relevant=[],
    ),

    # === ERA queries ===
    EvalQuery(
        query="classic literature from the 1800s",
        category="era",
        relevant=[],
    ),
    EvalQuery(
        query="modern thriller published after 2010",
        category="era",
        relevant=[],
    ),

    # === CROSS-DOMAIN queries (combine multiple signals) ===
    EvalQuery(
        query="feminist science fiction",
        category="cross-domain",
        relevant=[],
    ),
    EvalQuery(
        query="coming of age story set in India",
        category="cross-domain",
        relevant=[],
    ),
    EvalQuery(
        query="mathematical puzzles and recreational math",
        category="cross-domain",
        relevant=[],
    ),
    EvalQuery(
        query="true crime serial killer investigation",
        category="cross-domain",
        relevant=[],
    ),
    EvalQuery(
        query="graphic novel about war",
        category="cross-domain",
        relevant=[],
    ),
]


def get_eval_queries() -> list[EvalQuery]:
    """Return the evaluation query set."""
    return EVAL_QUERIES


def save_eval_dataset(path: Path = EVAL_DATA_PATH):
    """Save eval dataset to JSON for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(q) for q in EVAL_QUERIES]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} eval queries to {path}")


def load_eval_dataset(path: Path = EVAL_DATA_PATH) -> list[EvalQuery]:
    """Load eval dataset from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = []
    for item in data:
        q = EvalQuery(
            query=item["query"],
            source=item.get("source", "synthetic"),
            category=item.get("category", "topic"),
            relevant=[
                RelevanceJudgment(**r) for r in item.get("relevant", [])
            ],
        )
        queries.append(q)
    return queries


if __name__ == "__main__":
    save_eval_dataset()
    print(f"\nDataset stats:")
    print(f"  Total queries: {len(EVAL_QUERIES)}")
    print(f"  With judgments: {sum(1 for q in EVAL_QUERIES if q.relevant)}")
    print(f"  Without judgments (need manual): {sum(1 for q in EVAL_QUERIES if not q.relevant)}")
    print(f"\n  Categories:")
    from collections import Counter
    cats = Counter(q.category for q in EVAL_QUERIES)
    for cat, count in cats.most_common():
        print(f"    {cat}: {count}")
