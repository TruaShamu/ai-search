"""Validate a candidate replacement corpus before trusting it.

The v1 corpus acquired a data defect because an external description source was
joined on title alone and never audited afterwards. This script exists so the
replacement does not get the same free pass. It applies the same tests that
eventually exposed the v1 problem -- most importantly the cross-author
duplicate-description rate, which is directly comparable to the 26.5% measured
on the v1 Goodreads augmentation and the 0.5% measured on native OpenLibrary
records.

Usage:
    python scripts/validate_candidate_corpus.py
    python scripts/validate_candidate_corpus.py --dataset euclaise/goodreads_100k
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = REPO_ROOT / "data" / "eval" / "candidate_corpus_validation.json"

DEFAULT_DATASET = "euclaise/goodreads_100k"

BOILERPLATE = re.compile(
    r"^\s*(no description(?: available| provided)?|description not available|"
    r"no synopsis(?: available)?|currently unavailable|n/?a|none|tba|tbd)\s*[.!]*\s*$",
    re.IGNORECASE,
)

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "with", "at",
    "by", "from", "is", "was", "it", "its", "as", "that", "this", "be", "are",
    "her", "his", "she", "he", "they", "their", "but", "not", "have", "has",
}


def norm_title(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    out = []
    for ch in normalized:
        if unicodedata.combining(ch):
            continue
        cat = unicodedata.category(ch)
        out.append(" " if cat.startswith(("P", "S")) or ch.isspace() else ch.casefold())
    return re.sub(r"\s+", " ", "".join(out)).strip()


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", (text or "").casefold()) if w not in STOPWORDS and len(w) > 2}


def ascii_ratio(text: str) -> float:
    if not text:
        return 1.0
    return sum(1 for c in text if ord(c) < 128) / len(text)


def pct(n: int, d: int) -> str:
    return f"{(100.0 * n / d):.2f}%" if d else "n/a"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--split", default="train")
    ap.add_argument("--sample", type=int, default=0,
                    help="validate only the first N rows (0 = all)")
    args = ap.parse_args()

    from datasets import load_dataset

    print(f"Loading {args.dataset} ({args.split}) ...")
    ds = load_dataset(args.dataset, split=args.split)
    if args.sample:
        ds = ds.select(range(min(args.sample, len(ds))))
    n = len(ds)
    cols = list(ds.column_names)
    print(f"Rows: {n:,}")
    print(f"Columns: {cols}")

    # Map this dataset's column names onto the fields the corpus needs.
    def pick(*candidates: str) -> str | None:
        for c in candidates:
            if c in cols:
                return c
        return None

    c_title = pick("title", "name", "book_title")
    c_author = pick("author", "authors", "author_name")
    c_desc = pick("desc", "description", "summary", "synopsis")
    c_isbn = pick("isbn", "isbn13", "ISBN")
    c_genre = pick("genre", "genres", "categories")
    print(f"\nField mapping: title={c_title!r} author={c_author!r} "
          f"desc={c_desc!r} isbn={c_isbn!r} genre={c_genre!r}")

    missing = [nm for nm, c in
               (("title", c_title), ("author", c_author), ("description", c_desc)) if c is None]
    if missing:
        print(f"FATAL: dataset lacks required field(s): {missing}")
        return 1

    report: dict = {"dataset": args.dataset, "rows": n, "columns": cols,
                    "field_mapping": {"title": c_title, "author": c_author,
                                      "description": c_desc, "isbn": c_isbn, "genre": c_genre}}

    titles = ds[c_title]
    authors = ds[c_author]
    descs = ds[c_desc]
    isbns = ds[c_isbn] if c_isbn else [None] * n

    # ---- completeness -------------------------------------------------
    def empty(v) -> bool:
        return v is None or (isinstance(v, str) and not v.strip())

    n_no_title = sum(1 for v in titles if empty(v))
    n_no_author = sum(1 for v in authors if empty(v))
    n_no_desc = sum(1 for v in descs if empty(v))
    n_boiler = sum(1 for v in descs if isinstance(v, str) and BOILERPLATE.match(v))
    n_isbn = sum(1 for v in isbns if not empty(v))

    print("\n=== COMPLETENESS ===")
    print(f"  missing title       : {n_no_title:,} ({pct(n_no_title, n)})")
    print(f"  missing author      : {n_no_author:,} ({pct(n_no_author, n)})")
    print(f"  missing description : {n_no_desc:,} ({pct(n_no_desc, n)})")
    print(f"  boilerplate desc    : {n_boiler:,} ({pct(n_boiler, n)})")
    print(f"  has ISBN            : {n_isbn:,} ({pct(n_isbn, n)})")
    report["completeness"] = {
        "missing_title": n_no_title, "missing_author": n_no_author,
        "missing_description": n_no_desc, "boilerplate_description": n_boiler,
        "has_isbn": n_isbn,
    }

    # ---- usable rows ---------------------------------------------------
    usable = [
        i for i in range(n)
        if not empty(titles[i]) and not empty(authors[i]) and not empty(descs[i])
        and not BOILERPLATE.match(descs[i]) and len(descs[i].strip()) >= 30
    ]
    print(f"\n  USABLE ROWS (title+author+desc, desc>=30 chars): {len(usable):,} ({pct(len(usable), n)})")
    report["usable_rows"] = len(usable)

    # ---- THE KEY TEST: cross-author duplicate descriptions -------------
    # This is the signature that exposed the v1 defect. A description shared by
    # two books with different authors means at least one of them is wrong.
    by_desc: dict[str, set[str]] = defaultdict(set)
    for i in usable:
        by_desc[descs[i].strip()].add(norm_title(authors[i]))
    shared = {d: a for d, a in by_desc.items() if len(a) > 1}
    n_rows_shared = sum(1 for i in usable if descs[i].strip() in shared)

    print("\n=== CROSS-AUTHOR DUPLICATE DESCRIPTIONS (the v1 failure signature) ===")
    print(f"  distinct descriptions used by >1 author : {len(shared):,}")
    print(f"  rows carrying such a description        : {n_rows_shared:,} ({pct(n_rows_shared, len(usable))})")
    print("  v1 comparison -> Goodreads-augmented 26.50% | OpenLibrary-native 0.50%")
    report["cross_author_duplicate_descriptions"] = {
        "distinct_shared_descriptions": len(shared),
        "rows_affected": n_rows_shared,
        "rate": round(n_rows_shared / max(1, len(usable)), 4),
        "v1_goodreads_rate": 0.265,
        "v1_openlibrary_rate": 0.005,
    }

    # ---- duplicate titles ---------------------------------------------
    t_counts = Counter(norm_title(titles[i]) for i in usable)
    dup_titles = {t: c for t, c in t_counts.items() if c > 1}
    ta_counts = Counter((norm_title(titles[i]), norm_title(authors[i])) for i in usable)
    dup_ta = {k: c for k, c in ta_counts.items() if c > 1}
    print("\n=== DUPLICATES ===")
    print(f"  duplicate titles          : {len(dup_titles):,} titles, "
          f"{sum(dup_titles.values()):,} rows ({pct(sum(dup_titles.values()), len(usable))})")
    print(f"  duplicate (title, author) : {len(dup_ta):,} pairs, "
          f"{sum(dup_ta.values()):,} rows ({pct(sum(dup_ta.values()), len(usable))})")
    print("  (duplicate titles are harmless here -- author disambiguates them, "
          "which is the entire point of a single-source join)")
    report["duplicates"] = {
        "duplicate_title_rows": sum(dup_titles.values()),
        "duplicate_title_author_rows": sum(dup_ta.values()),
    }

    # ---- description/title coherence -----------------------------------
    zero_overlap = 0
    checked = 0
    for i in usable[:20000]:
        tw = words(titles[i])
        if not tw:
            continue
        checked += 1
        if not (tw & words(descs[i])):
            zero_overlap += 1
    print("\n=== TITLE/DESCRIPTION VOCABULARY OVERLAP (first 20k usable) ===")
    print(f"  zero-overlap rate : {pct(zero_overlap, checked)}  (checked {checked:,})")
    print("  v1 comparison -> clean OL control 15.5% | provably corrupt 25.2% | unverifiable 22.9%")
    report["zero_vocab_overlap_rate"] = round(zero_overlap / max(1, checked), 4)

    # ---- description length + language ---------------------------------
    lens = sorted(len(descs[i].strip()) for i in usable)
    if lens:
        med = lens[len(lens) // 2]
        p10, p90 = lens[int(0.1 * len(lens))], lens[int(0.9 * len(lens))]
        print("\n=== DESCRIPTION LENGTH (chars) ===")
        print(f"  p10={p10}  median={med}  p90={p90}  max={lens[-1]}")
        report["description_length"] = {"p10": p10, "median": med, "p90": p90, "max": lens[-1]}

    non_ascii = sum(1 for i in usable[:20000] if ascii_ratio(descs[i]) < 0.80)
    print(f"\n  mostly-non-ASCII descriptions (first 20k): {pct(non_ascii, min(20000, len(usable)))}")
    report["mostly_non_ascii_rate"] = round(non_ascii / max(1, min(20000, len(usable))), 4)

    # ---- eyeball a few -------------------------------------------------
    print("\n=== SAMPLE ROWS ===")
    for i in usable[:3]:
        print(f"  title : {titles[i][:70]}")
        print(f"  author: {authors[i][:70]}")
        print(f"  desc  : {descs[i].strip()[:150]}...")
        print()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved -> {OUT_FILE}")

    verdict_ok = (
        report["cross_author_duplicate_descriptions"]["rate"] < 0.05
        and report["zero_vocab_overlap_rate"] < 0.20
        and len(usable) > 20000
    )
    print(f"\nVERDICT: {'PASS' if verdict_ok else 'REVIEW NEEDED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
