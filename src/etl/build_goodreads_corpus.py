"""Build the v2 corpus from a single-source Goodreads dataset.

The v1 corpus joined descriptions onto OpenLibrary records by normalized title,
because the description source had no author column. That join is unverifiable
in principle, and 12.8% of the resulting index provably carried a description
belonging to a different author's book.

This builder takes title, author, and description from the *same row* of one
dataset, so a description can never be attached to the wrong book. The failure
mode is not mitigated, it is unrepresentable.

Usage:
    python -m src.etl.build_goodreads_corpus --limit 30000
    python -m src.etl.build_goodreads_corpus            # all usable rows
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import ftfy
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

from src.etl.clean_descriptions import clean_description

# langdetect is randomised; pin it so corpus builds are reproducible.
DetectorFactory.seed = 42

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT_DIR / "data" / "processed" / "books_goodreads_v2.jsonl"

DATASET = "euclaise/goodreads_100k"
SPLIT = "train"
MIN_DESCRIPTION_LENGTH = 30
MAX_SUBJECTS = 12

BOILERPLATE = re.compile(
    r"^\s*(no description(?: available| provided)?|description not available|"
    r"no synopsis(?: available)?|currently unavailable|n/?a|none|tba|tbd)\s*[.!]*\s*$",
    re.IGNORECASE,
)


def is_english(text: str) -> bool:
    """Whether a record's text is English.

    Both the embedding model (nomic-embed-text-v1.5) and the cross-encoder
    reranker are English-first. Measured on this stack, a translated sentence
    scores ~0.54-0.64 cosine against its English original where an English
    paraphrase scores ~0.79 -- above the ~0.27 unrelated baseline, but clearly
    degraded. Non-English books are therefore systematically harder to retrieve,
    and because the eval generates queries from corpus text they also produce
    queries the stack cannot serve, which depresses metrics for reasons that
    have nothing to do with retrieval design. Indexing only what the models can
    represent is the honest choice; the alternative is a multilingual model.
    """
    try:
        return detect(text[:600]) == "en"
    except LangDetectException:
        return False


def repair_text(value: str | None) -> str:
    """Undo mojibake in source text.

    26% of rows in the upstream dataset were encoded as UTF-8 and then decoded
    as cp1252, so "Aujourd'hui" arrives as "Aujourdâ€™hui" and "émir" as
    "Ã©mir". Left alone this reaches both the embedding model and the UI. ftfy
    detects and reverses the round-trip, and leaves already-correct text alone.
    """
    if not value:
        return ""
    return ftfy.fix_text(value)


def split_list(value: str | None, limit: int) -> list[str]:
    """Split a comma-delimited field into a trimmed, de-duplicated list.

    The source `genre` strings are themselves sometimes truncated mid-term with
    an ellipsis ("North American Hi..."), which would otherwise be indexed as a
    subject. Those fragments are dropped rather than stored.
    """
    if not value:
        return []
    seen: list[str] = []
    for piece in repair_text(value).split(","):
        item = piece.strip()
        if not item or "..." in item or item.endswith("…"):
            continue
        if item not in seen:
            seen.append(item)
        if len(seen) >= limit:
            break
    return seen


def is_usable(row: dict) -> bool:
    title = (row.get("title") or "").strip()
    author = (row.get("author") or "").strip()
    desc = (row.get("desc") or "").strip()
    if not title or not author or not desc:
        return False
    if len(desc) < MIN_DESCRIPTION_LENGTH:
        return False
    return not BOILERPLATE.match(desc)


def build_record(row: dict, idx: int) -> dict:
    isbn = (row.get("isbn") or "").strip() or None
    description = clean_description(repair_text(row.get("desc")).strip())

    return {
        # Namespaced so a v2 id can never be confused with an OpenLibrary work id.
        "work_id": f"gr:{isbn}" if isbn else f"gr:idx{idx}",
        "title": repair_text(row.get("title")).strip(),
        "authors": split_list(row.get("author"), limit=8),
        "description": description,
        "subjects": split_list(row.get("genre"), limit=MAX_SUBJECTS),
        # This dataset carries no publication year. The API already documents
        # `year` as unpopulated for the whole index, so this is status quo.
        "first_publish_year": None,
        "cover_id": None,
        "isbn": isbn,
        "subject_places": [],
        "subject_people": [],
        "subject_times": [],
        "tier": 1,
        "description_source": "goodreads_100k",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--limit", type=int, default=0,
                    help="sample this many usable rows (0 = keep all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-non-english", action="store_true",
                    help="index all languages (default: English only, matching the models)")
    args = ap.parse_args()

    from datasets import load_dataset

    print(f"Loading {DATASET} ...")
    ds = load_dataset(DATASET, split=SPLIT)
    print(f"  {len(ds):,} raw rows")

    usable = [i for i, row in enumerate(ds) if is_usable(row)]
    print(f"  {len(usable):,} usable rows (title + author + description)")

    if args.limit and args.limit < len(usable):
        rng = random.Random(args.seed)
        usable = sorted(rng.sample(usable, args.limit))
        print(f"  sampled {len(usable):,} rows (seed={args.seed})")

    if not args.keep_non_english:
        before = len(usable)
        # Detect on repaired text: mojibake confuses language detection.
        usable = [
            i for i in usable
            if is_english(repair_text(ds[i]["title"]) + " " + repair_text(ds[i]["desc"]))
        ]
        print(f"  {len(usable):,} English rows (dropped {before - len(usable):,} non-English)")

    # A description repeated across different authors means at least one of the
    # two is wrong. v1 shipped 26.5% of these; refusing them here keeps the
    # guarantee this migration exists to make.
    by_desc: dict[str, set[str]] = {}
    for i in usable:
        row = ds[i]
        by_desc.setdefault(row["desc"].strip(), set()).add((row["author"] or "").strip().casefold())
    ambiguous = {d for d, authors in by_desc.items() if len(authors) > 1}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    dropped_ambiguous = 0
    seen_ids: set[str] = set()

    with args.output.open("w", encoding="utf-8") as f:
        for idx in usable:
            row = ds[idx]
            if row["desc"].strip() in ambiguous:
                dropped_ambiguous += 1
                continue
            record = build_record(row, idx)
            # Guard against duplicate ISBNs producing colliding work_ids.
            if record["work_id"] in seen_ids:
                record["work_id"] = f"{record['work_id']}-{idx}"
            seen_ids.add(record["work_id"])
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"\n  dropped {dropped_ambiguous:,} rows sharing a description across authors")
    print(f"  wrote {written:,} records -> {args.output}")
    print(f"  size: {args.output.stat().st_size / 1024 / 1024:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
