"""Auto-annotate eval dataset using pooled results.

Strategy: Run each query through all retrieval modes, pool the unique results,
then use heuristic + title/description matching to assign relevance scores.
This gives us broader coverage than manually pre-assigning specific work_ids.
"""

import json
from pathlib import Path
from collections import defaultdict

from src.azure_search.search import HybridSearchEngine
from src.eval.dataset import get_eval_queries, EvalQuery, RelevanceJudgment


def pool_results(engine: HybridSearchEngine, queries: list[EvalQuery], top_k: int = 10):
    """Run each query through all modes and pool unique results."""
    pooled = {}  # query -> {doc_id: best_result_dict}

    modes = ["keyword", "vector", "hybrid"]

    for eq in queries:
        results_pool = {}
        for mode in modes:
            result = engine.search(query=eq.query, top_k=top_k, mode=mode)
            if "error" in result:
                continue
            for r in result["results"]:
                doc_id = r["id"]
                if doc_id not in results_pool:
                    results_pool[doc_id] = r

        pooled[eq.query] = results_pool
        print(f"  \"{eq.query}\" -> {len(results_pool)} unique docs pooled")

    return pooled


def heuristic_relevance(query: str, doc: dict) -> int:
    """Simple heuristic relevance scoring based on text overlap.
    
    Returns:
        0 = not relevant
        1 = partially relevant  
        2 = highly relevant
    """
    query_terms = set(query.lower().split())
    
    # Remove stop words
    stop_words = {"a", "an", "the", "of", "and", "in", "for", "to", "about", "from", "with", "books", "book", "novel", "story", "stories"}
    query_terms -= stop_words
    
    # Build doc text
    title = (doc.get("title") or "").lower()
    desc = (doc.get("description") or "").lower()
    subjects = " ".join(doc.get("subjects") or []).lower()
    authors = (doc.get("authors") or "").lower()
    doc_text = f"{title} {desc} {subjects} {authors}"
    
    # Count query term matches
    matches = sum(1 for term in query_terms if term in doc_text)
    match_ratio = matches / len(query_terms) if query_terms else 0
    
    # Check for strong signals
    # Title contains multiple query terms → likely relevant
    title_matches = sum(1 for term in query_terms if term in title)
    
    if title_matches >= 2 or match_ratio >= 0.7:
        return 2
    elif title_matches >= 1 or match_ratio >= 0.4:
        return 1
    else:
        return 0


def annotate_dataset(top_k: int = 10):
    """Auto-annotate the eval dataset with pooled + heuristic judgments."""
    queries = get_eval_queries()
    
    print("Loading search engine...")
    engine = HybridSearchEngine()
    
    print(f"\nPooling results for {len(queries)} queries...")
    pooled = pool_results(engine, queries, top_k=top_k)
    
    print("\nAnnotating with heuristic relevance...")
    annotated_queries = []
    
    for eq in queries:
        # Start with existing manual judgments
        existing_ids = {j.work_id for j in eq.relevant}
        judgments = list(eq.relevant)
        
        # Add heuristic judgments for pooled results
        if eq.query in pooled:
            for doc_id, doc in pooled[eq.query].items():
                if doc_id in existing_ids:
                    continue
                rel = heuristic_relevance(eq.query, doc)
                if rel > 0:
                    judgments.append(RelevanceJudgment(
                        work_id=doc_id,
                        relevance=rel,
                        title=doc.get("title", ""),
                    ))
        
        annotated_queries.append(EvalQuery(
            query=eq.query,
            relevant=judgments,
            source=eq.source,
            category=eq.category,
        ))
        
        new_count = len(judgments) - len(eq.relevant)
        if new_count > 0:
            print(f"  \"{eq.query}\": +{new_count} auto-judged (total: {len(judgments)})")
    
    # Save annotated dataset
    output_path = Path("data/eval/queries_annotated.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    from dataclasses import asdict
    data = [asdict(q) for q in annotated_queries]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Stats
    total_judgments = sum(len(q.relevant) for q in annotated_queries)
    queries_with = sum(1 for q in annotated_queries if q.relevant)
    print(f"\nAnnotation complete:")
    print(f"  Queries with judgments: {queries_with}/{len(annotated_queries)}")
    print(f"  Total judgments: {total_judgments}")
    print(f"  Saved to: {output_path}")
    
    return annotated_queries


if __name__ == "__main__":
    annotate_dataset()
