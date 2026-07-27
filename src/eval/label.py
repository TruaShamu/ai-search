"""Human labeling CLI -- validate the LLM judge against your own judgment.

The eval pipeline's relevance labels come from an LLM. Nothing has ever checked
that those labels track human judgment. This tool samples query/document pairs,
hides the machine's grade, and asks you to grade them yourself. The result is a
second judgment file that can be scored against the LLM's with Cohen's kappa:

    python -m src.eval.label --n 50
    python -m src.eval.judge --agreement data/eval/v2/judgments.json \
                                         data/eval/v2/human.json

Design notes:
  - The LLM grade is never displayed during labeling. Seeing it first would
    anchor the labeler and make the agreement score meaningless.
  - Sampling defaults to stratified across the LLM's grades so that rare
    grade-2 pairs actually appear. This is deliberately *not* a representative
    sample -- see the warning printed on completion.
  - Progress is saved after every keypress, so the session is resumable.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

DATA_DIR = Path("data/eval/v2")
JUDGMENTS_PATH = DATA_DIR / "judgments.json"
POOLED_PATH = DATA_DIR / "pooled.json"
HUMAN_PATH = DATA_DIR / "human.json"

GRADE_HELP = {
    0: "irrelevant  - does not answer the query",
    1: "partial     - related, but not what was asked for",
    2: "highly      - a direct, satisfying answer",
}


# ---------------------------------------------------------------- console io

def _enable_utf8_stdout() -> None:
    """Best-effort UTF-8 console output.

    Descriptions contain accented and non-Latin text that a legacy Windows
    code page renders as '?'. The labeler has to actually read these to grade
    them, so try to upgrade the stream; _safe() still covers the failure case.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _safe(text: str) -> str:
    """Make text printable on a legacy Windows code page."""
    enc = sys.stdout.encoding or "utf-8"
    return str(text).encode(enc, "replace").decode(enc)


def _read_key() -> str:
    """Read a single keypress without requiring Enter, where possible."""
    try:
        import msvcrt  # Windows

        ch = msvcrt.getch()
        if ch in (b"\x00", b"\xe0"):  # function/arrow key prefix
            msvcrt.getch()
            return ""
        return ch.decode("utf-8", "ignore").lower()
    except ImportError:
        pass

    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1).lower()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        return (sys.stdin.readline().strip() or " ")[:1].lower()


def _wrap(text: str, width: int, indent: str, max_lines: int) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(indent + cur)
            cur = w
            if len(lines) == max_lines:
                lines[-1] = lines[-1].rstrip() + " ..."
                return lines
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(indent + cur)
    return lines


# ------------------------------------------------------------------ sampling

def _grade_of(entry: dict) -> Optional[int]:
    """Read a grade from either the `grade` or `relevance` field."""
    g = entry.get("grade")
    if g is None:
        g = entry.get("relevance")
    return g


def select_pairs(
    judgments: dict[str, list[dict]],
    pooled: dict[str, list[dict]],
    n: int,
    seed: int,
    strategy: str,
) -> list[dict]:
    """Pick n query/doc pairs to label, enriched with document metadata."""
    doc_index = {
        query: {d["work_id"]: d for d in docs} for query, docs in pooled.items()
    }

    candidates: list[dict] = []
    for query, entries in judgments.items():
        docs = doc_index.get(query, {})
        for e in entries:
            grade = _grade_of(e)
            doc = docs.get(e["work_id"])
            if grade is None or doc is None:
                continue
            candidates.append(
                {
                    "query": query,
                    "work_id": e["work_id"],
                    "title": doc.get("title") or e.get("title") or "(untitled)",
                    "authors": doc.get("authors") or "",
                    "subjects": doc.get("subjects") or [],
                    "description": doc.get("description") or "",
                    "llm_grade": grade,
                }
            )

    rng = random.Random(seed)
    rng.shuffle(candidates)

    if strategy == "random":
        chosen = candidates[:n]
    else:
        buckets: dict[int, list[dict]] = {0: [], 1: [], 2: []}
        for c in candidates:
            buckets.setdefault(c["llm_grade"], []).append(c)
        chosen = []
        # round-robin across grades so rare grade-2 pairs are represented
        while len(chosen) < n and any(buckets.values()):
            for g in (0, 1, 2):
                if buckets.get(g) and len(chosen) < n:
                    chosen.append(buckets[g].pop())

    rng.shuffle(chosen)  # don't present in grade order
    return chosen


# ----------------------------------------------------------------- rendering

def _render(pair: dict, idx: int, total: int) -> None:
    print("\n" + "=" * 74)
    print(f"[{idx}/{total}]  QUERY:  {_safe(pair['query'])}")
    print("=" * 74)
    print(f"\n  {_safe(pair['title'])}")
    if pair["authors"]:
        print(f"  by {_safe(pair['authors'])}")

    subjects = pair["subjects"]
    if subjects:
        joined = ", ".join(str(s) for s in subjects[:6])
        print(f"  subjects: {_safe(joined)}")

    desc = pair["description"]
    print()
    if desc:
        for line in _wrap(_safe(desc), 68, "  ", 6):
            print(line)
    else:
        print("  (no description available)")

    print("\n" + "-" * 74)
    for g, label in GRADE_HELP.items():
        print(f"   {g} = {label}")
    print("   s = skip    b = back    q = save & quit")
    print("-" * 74)


# ------------------------------------------------------------------ i/o

def load_progress(path: Path) -> dict[str, list[dict]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_progress(path: Path, data: dict[str, list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _already_labeled(human: dict[str, list[dict]]) -> set[tuple[str, str]]:
    return {
        (q, e["work_id"]) for q, entries in human.items() for e in entries
    }


# ------------------------------------------------------------------ session

def run_labeling(pairs: list[dict], out_path: Path, strategy: str) -> None:
    human = load_progress(out_path)
    done = _already_labeled(human)

    todo = [p for p in pairs if (p["query"], p["work_id"]) not in done]
    if not todo:
        print(f"All {len(pairs)} sampled pairs are already labeled in {out_path}.")
        _summary(human, pairs, strategy, out_path)
        return

    if done:
        print(f"Resuming: {len(pairs) - len(todo)} already labeled, {len(todo)} to go.")

    print(
        "\nGrade each book against the query. The LLM's grade is hidden on "
        "purpose --\njudge it yourself, then we compare."
    )

    i = 0
    while i < len(todo):
        pair = todo[i]
        _render(pair, i + 1, len(todo))
        key = ""
        while key not in {"0", "1", "2", "s", "b", "q"}:
            key = _read_key()

        if key == "q":
            break
        if key == "b":
            if i > 0:
                i -= 1
                prev = todo[i]
                entries = human.get(prev["query"], [])
                human[prev["query"]] = [
                    e for e in entries if e["work_id"] != prev["work_id"]
                ]
                if not human[prev["query"]]:
                    del human[prev["query"]]
                save_progress(out_path, human)
            continue
        if key == "s":
            i += 1
            continue

        grade = int(key)
        human.setdefault(pair["query"], []).append(
            {
                "work_id": pair["work_id"],
                "title": pair["title"],
                # both keys: `grade` for judge.py, `relevance` for metrics.py
                "grade": grade,
                "relevance": grade,
            }
        )
        save_progress(out_path, human)
        i += 1

    _summary(human, pairs, strategy, out_path)


def _summary(
    human: dict[str, list[dict]],
    pairs: list[dict],
    strategy: str,
    out_path: Path,
) -> None:
    labeled = sum(len(v) for v in human.values())
    dist = Counter(_grade_of(e) for v in human.values() for e in v)

    print("\n" + "=" * 74)
    print(f"Labeled {labeled} pairs -> {out_path}")
    print(
        f"  your grades: 0={dist.get(0, 0)}  1={dist.get(1, 0)}  2={dist.get(2, 0)}"
    )

    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "sampling_strategy": strategy,
                "n_sampled": len(pairs),
                "n_labeled": labeled,
                "note": (
                    "Stratified sampling over-represents rare grades. Cohen's "
                    "kappa depends on class prevalence, so a kappa computed on "
                    "a stratified sample is not directly comparable to one from "
                    "a representative sample. Report the strategy alongside it."
                )
                if strategy == "stratified"
                else "Representative random sample of judged pairs.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if strategy == "stratified":
        print(
            "\n  NOTE: this was a stratified sample (rare grades over-represented)."
            "\n  Kappa depends on class prevalence -- always report the sampling"
            "\n  strategy with the number. Use --strategy random for a"
            "\n  representative estimate."
        )

    print("\nNext:")
    print(f"  python -m src.eval.judge --agreement {JUDGMENTS_PATH} {out_path}")
    print("=" * 74)


# --------------------------------------------------------------------- cli

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Hand-label query/document pairs to validate the LLM judge."
    )
    p.add_argument("--n", type=int, default=50, help="pairs to sample (default 50)")
    p.add_argument("--seed", type=int, default=42, help="sampling seed")
    p.add_argument(
        "--strategy",
        choices=["stratified", "random"],
        default="stratified",
        help="stratified (default) surfaces rare grade-2 pairs; "
        "random gives a prevalence-representative kappa",
    )
    p.add_argument("--judgments", type=Path, default=JUDGMENTS_PATH)
    p.add_argument("--pooled", type=Path, default=POOLED_PATH)
    p.add_argument("--out", type=Path, default=HUMAN_PATH)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="show the sample and exit without prompting",
    )
    args = p.parse_args(argv)

    _enable_utf8_stdout()

    for path in (args.judgments, args.pooled):
        if not path.exists():
            p.error(
                f"{path} not found. Run the eval pipeline first "
                f"(python scripts/eval_redesign.py --step pool)."
            )

    judgments = json.loads(args.judgments.read_text(encoding="utf-8"))
    pooled = json.loads(args.pooled.read_text(encoding="utf-8"))

    pairs = select_pairs(judgments, pooled, args.n, args.seed, args.strategy)
    if not pairs:
        p.error("No overlapping pairs between judgments and pooled results.")

    if args.dry_run:
        hidden = Counter(x["llm_grade"] for x in pairs)
        print(f"Sampled {len(pairs)} pairs ({args.strategy}).")
        print(f"  hidden LLM grade distribution: {dict(sorted(hidden.items()))}")
        for x in pairs[:5]:
            print(f"  - {_safe(x['query'])[:45]:45} | {_safe(x['title'])[:30]}")
        return

    run_labeling(pairs, args.out, args.strategy)


if __name__ == "__main__":
    main()
