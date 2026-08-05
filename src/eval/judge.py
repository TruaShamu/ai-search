"""Calibrated LLM-as-judge for book-search relevance with quality measurement.

Improvements over the prior judge (llm_judge.py):
  - Few-shot calibration with worked grade-boundary examples
  - Self-consistency via k samples + majority vote
  - Verbosity / position bias mitigation
  - Grade distribution monitoring with automatic warnings
  - Cohen's kappa & self-consistency metrics
  - Gold-doc agreement scoring (corpus-grounded queries)
  - Contradiction detection (reasoning vs. grade)
  - CSV export of disagreements for human audit
  - Full document view (no aggressive truncation)
  - Caching, retry w/ backoff, fail-fast on 4xx, None for failures

Usage:
    python -m src.eval.judge --dry-run                 # offline smoke-test
    python -m src.eval.judge --input data/eval/v2/pooled.json
    python -m src.eval.judge --input data/eval/v2/pooled.json --k 5
    python -m src.eval.judge --agreement ref_a.json ref_b.json
    python -m src.eval.judge --gold-check queries.json judgments.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from src.eval.agreement import (
    cohens_kappa,
    compute_agreement,
    contradiction_report,
    gold_doc_agreement,
    grade_distribution,
    krippendorffs_alpha,
    self_consistency_report,
)

# Re-exported for tests
__all__ = [
    "cohens_kappa",
    "krippendorffs_alpha",
]
from src.eval.llm_client import AzureOpenAIClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/eval/v2")
CACHE_PATH = DATA_DIR / "judge_cache.json"
JUDGMENTS_OUT = DATA_DIR / "judgments_v2.json"
CSV_OUT = DATA_DIR / "judgments_audit.csv"
# Contradiction-detection keywords
CONTRADICTION_NEGATIVE_WORDS = [
    "not relevant", "unrelated", "no connection", "nothing to do",
    "does not match", "wrong topic", "irrelevant", "no overlap",
    "not what the user", "not a match", "misleading",
]
CONTRADICTION_POSITIVE_WORDS = [
    "exactly what", "perfect match", "directly relevant",
    "precisely what", "ideal result", "highly relevant",
    "exactly matches",
]

# ---------------------------------------------------------------------------
# Few-shot calibrated system prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a **book-search relevance judge**. A user typed a search query into a
book-search engine and the system returned a candidate book. Your job is to
decide how well the book satisfies the user's information need.

### Grading scale

| Grade | Meaning | Rule of thumb |
|-------|---------|---------------|
| **0 — NOT RELEVANT** | The book has no meaningful connection to the query intent. A keyword happens to overlap, or the topic is completely wrong. | The user would feel the result was noise. |
| **1 — PARTIALLY RELEVANT** | The book is on a *related* topic or shares a theme, but it is NOT what the user was most likely looking for. | The user *might* skim it but would keep searching. |
| **2 — HIGHLY RELEVANT** | The book directly addresses the query intent. A user issuing this query would be satisfied to find this book. | The user would click, borrow, or buy it. |

### Calibration examples

**Example A — Grade 0 (keyword overlap, wrong topic)**
Query: "python programming"
Book: *Monty Python's Flying Circus: Complete & Utter Scripts*
Reasoning: The word "python" appears, but the book is about comedy television,
not programming. A user searching for a programming book would consider this
noise.
Grade: 0

**Example B — Grade 1 (adjacent topic, not the target)**
Query: "machine learning for beginners"
Book: *Introduction to Statistics and Data Analysis*
Reasoning: Statistics is foundational to machine learning, so the book is
*related*, but a user who specifically wants an ML book would keep looking.
Grade: 1

**Example C — Grade 2 (direct match)**
Query: "machine learning for beginners"
Book: *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*
Reasoning: This is an introductory ML book with hands-on exercises — exactly
what the query asks for.
Grade: 2

**Example D — Grade 0 (superficially similar, wrong audience)**
Query: "children's bedtime stories"
Book: *The Interpretation of Dreams* by Sigmund Freud
Reasoning: Although the title mentions "dreams," this is a psychoanalytic text,
not a children's book. A parent looking for bedtime stories would discard it.
Grade: 0

**Example E — Grade 1 (right genre, wrong scope)**
Query: "history of ancient Rome"
Book: *A History of the Ancient World* by Susan Wise Bauer
Reasoning: This covers ancient civilisations broadly, including Rome, but a
user who specifically wants Roman history would prefer a Rome-focused book.
Grade: 1

**Example F — Grade 2 (exploratory query satisfied)**
Query: "books about loneliness"
Book: *Eleanor Oliphant Is Completely Fine* by Gail Honeyman
Reasoning: The novel's central theme is loneliness and social isolation — it
directly speaks to the query.
Grade: 2

### Rules
1. Judge SEMANTIC relevance to the user's likely intent, not keyword overlap.
2. When the query is broad or exploratory, be **generous with grade 1** but
   reserve grade 2 for books clearly within the core intent.
3. A long description does NOT make a book more relevant — judge the *topic
   match*, not the amount of text.
4. Respond in **exactly** this JSON format and nothing else:

```json
{"grade": <0|1|2>, "reasoning": "<one concise sentence explaining your grade>"}
```
"""

# User message template — fields presented in randomised order to reduce
# position bias (the code shuffles the field block at call time).
_FIELD_TEMPLATES = {
    "title": "Title: {title}",
    "authors": "Author(s): {authors}",
    "subjects": "Subjects: {subjects}",
    "description": "Description:\n{description}",
}


def _build_user_message(query: str, doc: dict, *, shuffle_fields: bool = True) -> str:
    """Render the user-turn message, optionally shuffling field order."""
    title = doc.get("title", "Unknown")
    authors = doc.get("authors", "Unknown")
    subjects = ", ".join(doc.get("subjects", [])[:15]) or "None listed"
    description = doc.get("description") or "No description available."

    # Full description — do NOT truncate.  The retriever saw the full text;
    # the judge should too.  Corpus p90 ≈ 1 300 chars, well within context.
    field_values = {
        "title": title,
        "authors": authors,
        "subjects": subjects,
        "description": description,
    }

    keys = list(_FIELD_TEMPLATES.keys())
    if shuffle_fields:
        random.shuffle(keys)

    field_block = "\n".join(
        _FIELD_TEMPLATES[k].format(**{k: field_values[k]}) for k in keys
    )

    return f'Query: "{query}"\n\nBook:\n{field_block}'


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SampleResult:
    """One LLM sample for a (query, doc) pair."""
    grade: Optional[int]
    reasoning: str
    raw: str = ""


@dataclass
class JudgmentResult:
    """Aggregated judgment for a (query, doc) pair."""
    query: str
    work_id: str
    title: str
    grade: Optional[int]        # majority-vote grade, None if unjudged/no consensus
    reasoning: str              # reasoning from the majority sample
    samples: list[int]          # individual sample grades (excl. failures)
    agreement: float            # fraction of k_requested that agreed with majority
    k_requested: int = 3        # how many samples were requested
    n_samples_ok: int = 0       # how many samples returned a valid grade
    low_confidence: bool = False  # True if no consensus or < 2 successful samples
    contradiction: bool = False # reasoning contradicts grade?
    contradiction_detail: str = ""


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def _pair_key(query: str, work_id: str) -> str:
    """Stable hash key for a (query, work_id) pair."""
    raw = f"{query.strip().lower()}||{work_id.strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_cache(path: Path = CACHE_PATH) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_cache(cache: dict, path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Shared LLM client
# ---------------------------------------------------------------------------
AzureJudgeClient = AzureOpenAIClient


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
_GRADE_RE = re.compile(r'"grade"\s*:\s*([012])')


def _parse_judge_response(raw: str) -> tuple[Optional[int], str]:
    """Extract (grade, reasoning) from the JSON response. Returns (None, '')
    if parsing fails — never coerces a failure to 0."""
    if not raw:
        return None, ""
    # Strip markdown fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    try:
        obj = json.loads(cleaned)
        grade = int(obj.get("grade", -1))
        if grade not in (0, 1, 2):
            return None, obj.get("reasoning", "")
        return grade, obj.get("reasoning", "")
    except (json.JSONDecodeError, ValueError, TypeError):
        # Fallback: regex for grade
        m = _GRADE_RE.search(raw)
        if m:
            return int(m.group(1)), ""
        return None, ""


def _detect_contradiction(grade: int, reasoning: str) -> tuple[bool, str]:
    """Heuristic: flag when reasoning language contradicts the numeric grade."""
    low = reasoning.lower()
    if grade >= 2:
        for phrase in CONTRADICTION_NEGATIVE_WORDS:
            if phrase in low:
                return True, f'Grade {grade} but reasoning says "{phrase}"'
    if grade == 0:
        for phrase in CONTRADICTION_POSITIVE_WORDS:
            if phrase in low:
                return True, f'Grade {grade} but reasoning says "{phrase}"'
    return False, ""


# ---------------------------------------------------------------------------
# Core judging
# ---------------------------------------------------------------------------
def _has_strict_majority(samples: list[int]) -> tuple[bool, Optional[int], int]:
    """Check whether any grade has a strict majority (> half) of samples.

    Returns (has_majority, winning_grade_or_None, winning_count).
    For a genuine 2-way tie between non-zero grades, the conservative
    tie-break (lower grade) is applied and counts as a majority.
    A full 3-way split (all counts equal) is NOT a majority.
    """
    if not samples:
        return False, None, 0
    counts = Counter(samples)
    n = len(samples)
    max_count = max(counts.values())

    # Strict majority: one grade has more than half
    if max_count * 2 > n:
        # Unique winner
        winners = [g for g, c in counts.items() if c == max_count]
        return True, min(winners), max_count  # conservative tie-break among equals

    # 2-way tie (e.g. k=4 with 2-2 split): apply conservative tie-break
    # but only when there are exactly 2 candidates at the max count
    candidates = sorted(g for g, c in counts.items() if c == max_count)
    if len(candidates) == 2:
        return True, candidates[0], max_count  # lower grade wins 2-way ties

    # 3+ way tie (e.g. [2,1,0]) -- no majority
    return False, None, max_count


def _draw_samples(
    client: AzureJudgeClient,
    query: str,
    doc: dict,
    n: int,
    *,
    start_index: int,
    temperature: float,
    shuffle_fields: bool,
    dry_run: bool,
    rate_limit_sleep: float,
) -> tuple[list[int], list[str]]:
    """Draw n LLM samples and return (grades, reasonings) for successes."""
    grades: list[int] = []
    reasonings: list[str] = []
    work_id = doc.get("work_id", doc.get("id", ""))
    for i in range(n):
        idx = start_index + i
        if dry_run:
            synth_grade = hash(f"{query}|{work_id}|{idx}") % 3
            grades.append(synth_grade)
            reasonings.append(f"[dry-run] synthetic grade {synth_grade}")
            continue

        user_msg = _build_user_message(query, doc, shuffle_fields=shuffle_fields)
        raw = client.call(
            SYSTEM_PROMPT,
            user_msg,
            temperature=temperature if n > 1 else 0.0,
            max_tokens=120,
        )
        grade, reasoning = _parse_judge_response(raw or "")
        if grade is not None:
            grades.append(grade)
            reasonings.append(reasoning)

        if rate_limit_sleep > 0 and i < n - 1:
            time.sleep(rate_limit_sleep)

    return grades, reasonings


def judge_pair(
    client: AzureJudgeClient,
    query: str,
    doc: dict,
    *,
    k: int = 3,
    max_escalation: int = 2,
    temperature: float = 0.3,
    shuffle_fields: bool = True,
    dry_run: bool = False,
    rate_limit_sleep: float = 0.5,
) -> JudgmentResult:
    """Judge a single (query, doc) pair with k samples and majority vote.

    Tie-break rules (documented):
      - 2-way tie: pick the *lower* grade (conservative). Over-grading was
        the documented failure mode; conservative tie-break biases toward safety.
      - No majority (3+ way split): escalate by drawing up to
        ``max_escalation`` extra samples and re-voting. If still no majority,
        return grade=None with reasoning="no_consensus" and low_confidence=True.
        The pair is treated as *unjudged*, not as irrelevant.

    ``agreement`` is always computed relative to the *requested* sample count
    (k + any escalation draws), not the number of successful responses, so
    that a single surviving sample does not masquerade as 100 % consensus.

    A pair with fewer than 2 successful samples is always marked
    low_confidence regardless of the vote outcome, since self-consistency
    cannot be measured from a single observation.
    """
    work_id = doc.get("work_id", doc.get("id", ""))
    title = doc.get("title", "")

    # ---- initial k samples ----
    samples, reasonings = _draw_samples(
        client, query, doc, k,
        start_index=0,
        temperature=temperature,
        shuffle_fields=shuffle_fields,
        dry_run=dry_run,
        rate_limit_sleep=rate_limit_sleep,
    )

    total_requested = k

    if not samples:
        return JudgmentResult(
            query=query, work_id=work_id, title=title,
            grade=None, reasoning="all_samples_failed",
            samples=[], agreement=0.0,
            k_requested=k, n_samples_ok=0, low_confidence=True,
        )

    # ---- majority check with escalation ----
    has_maj, winner, win_count = _has_strict_majority(samples)

    if not has_maj and max_escalation > 0 and len(samples) >= 2:
        # Escalate: draw extra samples to try to break the split
        extra_grades, extra_reasonings = _draw_samples(
            client, query, doc, max_escalation,
            start_index=k,
            temperature=temperature,
            shuffle_fields=shuffle_fields,
            dry_run=dry_run,
            rate_limit_sleep=rate_limit_sleep,
        )
        samples.extend(extra_grades)
        reasonings.extend(extra_reasonings)
        total_requested += max_escalation
        has_maj, winner, win_count = _has_strict_majority(samples)

    n_ok = len(samples)

    # ---- resolve ----
    if not has_maj or winner is None:
        # Still no consensus after escalation
        return JudgmentResult(
            query=query, work_id=work_id, title=title,
            grade=None, reasoning="no_consensus",
            samples=samples, agreement=0.0,
            k_requested=total_requested, n_samples_ok=n_ok,
            low_confidence=True,
        )

    # Agreement relative to total_requested (not n_ok)
    agreement = win_count / total_requested

    majority_idx = samples.index(winner)
    reasoning = reasonings[majority_idx] if majority_idx < len(reasonings) else ""

    contra, contra_detail = _detect_contradiction(winner, reasoning)

    # Fewer than 2 successful samples -> low confidence even with a "winner"
    low_conf = n_ok < 2

    return JudgmentResult(
        query=query, work_id=work_id, title=title,
        grade=winner, reasoning=reasoning,
        samples=samples, agreement=agreement,
        k_requested=total_requested, n_samples_ok=n_ok,
        low_confidence=low_conf,
        contradiction=contra, contradiction_detail=contra_detail,
    )


# ---------------------------------------------------------------------------
# Batch judging with caching & incremental save
# ---------------------------------------------------------------------------
def judge_pooled(
    pooled: dict[str, list[dict]],
    *,
    k: int = 3,
    max_escalation: int = 2,
    temperature: float = 0.3,
    dry_run: bool = False,
    rate_limit_sleep: float = 0.5,
    save_every: int = 5,
) -> dict[str, list[dict]]:
    """Judge all (query, doc) pairs in a pooled dict.

    Returns {query: [{work_id, title, grade, reasoning, samples, agreement, ...}, ...]}.
    """
    client = AzureJudgeClient(dry_run=dry_run)
    cache = load_cache() if not dry_run else {}
    results: dict[str, list[dict]] = {}
    queries_done = 0

    total_pairs = sum(len(docs) for docs in pooled.values())
    print(f"Judging {total_pairs} pairs across {len(pooled)} queries (k={k}, dry_run={dry_run})")

    for qi, (query, docs) in enumerate(pooled.items()):
        query_results = []
        for doc in docs:
            wid = doc.get("work_id", doc.get("id", ""))
            pk = _pair_key(query, wid)

            # Check cache
            if pk in cache and not dry_run:
                query_results.append(cache[pk])
                continue

            jr = judge_pair(
                client, query, doc,
                k=k, max_escalation=max_escalation,
                temperature=temperature,
                dry_run=dry_run, rate_limit_sleep=rate_limit_sleep,
            )
            entry = asdict(jr)
            query_results.append(entry)
            if not dry_run:
                cache[pk] = entry

        results[query] = query_results
        queries_done += 1

        if queries_done % save_every == 0:
            if not dry_run:
                save_cache(cache)
            _save_judgments(results)
            print(f"  [{queries_done}/{len(pooled)}] saved checkpoint")

    # Final save
    if not dry_run:
        save_cache(cache)
    _save_judgments(results)

    if not dry_run:
        cost = client.cost_estimate()
        print(f"  API cost: ${cost['cost_usd']:.4f} ({cost['total_calls']} calls)")

    return results


def _save_judgments(results: dict, path: Path = JUDGMENTS_OUT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# CSV export for human audit
# ---------------------------------------------------------------------------
def export_audit_csv(results: dict[str, list[dict]], path: Path = CSV_OUT) -> Path:
    """Export a stratified sample biased toward disagreements and boundary cases.

    Sort priority (highest-value rows for human review first):
      1. no_consensus (grade=None from split votes) -- the pairs the judge
         could not resolve, most valuable for a human to break the tie
      2. low_confidence (< 2 successful samples, or escalation-resolved)
      3. contradictions (reasoning vs. grade mismatch)
      4. low agreement
      5. boundary grades (grade 1)
    """
    rows: list[dict] = []
    for query, jlist in results.items():
        for j in jlist:
            rows.append({
                "query": query,
                "work_id": j.get("work_id", ""),
                "title": j.get("title", ""),
                "grade": j.get("grade"),
                "reasoning": j.get("reasoning", ""),
                "samples": str(j.get("samples", [])),
                "k_requested": j.get("k_requested", ""),
                "n_samples_ok": j.get("n_samples_ok", ""),
                "agreement": j.get("agreement", ""),
                "low_confidence": j.get("low_confidence", False),
                "contradiction": j.get("contradiction", False),
                "contradiction_detail": j.get("contradiction_detail", ""),
            })

    def sort_key(r: dict) -> tuple:
        no_cons = 0 if r.get("grade") is None else 1
        low_conf = 0 if r.get("low_confidence") else 1
        contra = 0 if r.get("contradiction") else 1
        agr = r.get("agreement", 1.0) if isinstance(r.get("agreement"), (int, float)) else 1.0
        boundary = 0 if r.get("grade") == 1 else 1
        return (no_cons, low_conf, contra, agr, boundary)

    rows.sort(key=sort_key)

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query", "work_id", "title", "grade", "reasoning",
        "samples", "k_requested", "n_samples_ok", "agreement",
        "low_confidence", "contradiction", "contradiction_detail",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    no_cons_count = sum(1 for r in rows if r.get("grade") is None)
    low_conf_count = sum(1 for r in rows if r.get("low_confidence"))
    print(f"\nExported {len(rows)} rows to {path}")
    print(f"  no_consensus (top of file): {no_cons_count}")
    print(f"  low_confidence:             {low_conf_count}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Calibrated LLM-as-judge for book search relevance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", type=Path, default=DATA_DIR / "pooled.json",
                   help="Pooled (query→docs) JSON file")
    p.add_argument("--output", type=Path, default=JUDGMENTS_OUT,
                   help="Output judgments JSON file")
    p.add_argument("--csv", type=Path, default=CSV_OUT,
                   help="Output audit CSV file")
    p.add_argument("--k", type=int, default=3,
                   help="Number of samples per pair (default 3)")
    p.add_argument("--max-escalation-samples", type=int, default=2,
                   help="Extra samples to draw when initial k has no majority (default 2)")
    p.add_argument("--temperature", type=float, default=0.3,
                   help="Sampling temperature (default 0.3)")
    p.add_argument("--dry-run", action="store_true",
                   help="Exercise all logic without API calls")
    p.add_argument("--rate-limit", type=float, default=0.5,
                   help="Sleep between API calls in seconds")
    p.add_argument("--agreement", nargs=2, type=Path, metavar=("A", "B"),
                   help="Compute agreement between two judgment JSON files")
    p.add_argument("--gold-check", nargs=2, type=Path, metavar=("QUERIES", "JUDGMENTS"),
                   help="Score judge against gold_work_ids in queries file")
    return p


def _load_pooled(path: Path) -> dict[str, list[dict]]:
    """Load pooled data. Falls back to generating synthetic data for dry-run."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _make_synthetic_pooled(n_queries: int = 8, docs_per_query: int = 5) -> dict[str, list[dict]]:
    """Generate synthetic pooled data for dry-run testing."""
    queries = [
        "python programming tutorials",
        "romance novels set in Scotland",
        "history of ancient Rome",
        "children's bedtime stories",
        "machine learning for beginners",
        "world war 2 memoir",
        "cooking Italian food",
        "mystery detective novels",
    ][:n_queries]

    titles = [
        ("Monty Python's Flying Circus Scripts", "Terry Jones", "Comedy, Television"),
        ("Learning Python", "Mark Lutz", "Programming, Computers"),
        ("The Bride", "Julie Garwood", "Romance, Historical Fiction"),
        ("Seducing the Highlander", "Michele Sinclair", "Romance, Scotland"),
        ("SPQR: A History of Ancient Rome", "Mary Beard", "History, Rome"),
        ("Goodnight Moon", "Margaret Wise Brown", "Children, Bedtime"),
        ("Introduction to Statistics", "David Moore", "Statistics, Mathematics"),
        ("Hands-On Machine Learning", "Aurélien Géron", "ML, Programming"),
        ("The War-Time Diary", "G. Bathurst", "WWII, Memoir"),
        ("Celebration Food", "unknown", "Cooking, Italian"),
    ]

    pooled = {}
    for qi, q in enumerate(queries):
        docs = []
        for di in range(docs_per_query):
            ti = (qi * 3 + di) % len(titles)
            t, a, s = titles[ti]
            docs.append({
                "work_id": f"OL{qi*100+di}W",
                "title": t,
                "authors": a,
                "subjects": s.split(", "),
                "description": f"A book about {t.lower()}. " * 5,
            })
        pooled[q] = docs
    return pooled


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # ---- Agreement mode ---- #
    if args.agreement:
        a_data = json.loads(args.agreement[0].read_text(encoding="utf-8"))
        b_data = json.loads(args.agreement[1].read_text(encoding="utf-8"))
        compute_agreement(a_data, b_data)
        return

    # ---- Gold-check mode ---- #
    if args.gold_check:
        queries_path, judgments_path = args.gold_check
        jdata = json.loads(judgments_path.read_text(encoding="utf-8"))
        gold_doc_agreement(jdata, queries_path)
        return

    # ---- Judge mode ---- #
    global JUDGMENTS_OUT, CSV_OUT  # noqa: PLW0603
    JUDGMENTS_OUT = args.output
    CSV_OUT = args.csv

    pooled = _load_pooled(args.input)
    if not pooled:
        if args.dry_run:
            print("No pooled data found -- generating synthetic data for dry-run.")
            pooled = _make_synthetic_pooled()
        else:
            print(f"ERROR: pooled data not found at {args.input}")
            sys.exit(1)

    results = judge_pooled(
        pooled,
        k=args.k,
        max_escalation=args.max_escalation_samples,
        temperature=args.temperature,
        dry_run=args.dry_run,
        rate_limit_sleep=args.rate_limit,
    )

    # Reports
    grade_distribution(results)
    self_consistency_report(results)
    contradiction_report(results)

    # Gold-doc check (defensive — file may not exist yet)
    gold_candidates = [
        DATA_DIR / "queries.json",
        Path("data/eval/queries.json"),
    ]
    for gp in gold_candidates:
        gold_doc_agreement(results, gp)

    export_audit_csv(results, args.csv)

    print("\n[OK] Judge run complete.")


if __name__ == "__main__":
    main()
