"""
ETL Pipeline — Extract, clean, and export OpenLibrary works for search indexing.

Reads works + authors parquet from HuggingFace cache, joins author names,
cleans text, assigns quality tiers, and exports to JSONL.
"""

import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.etl.clean import (
    extract_year,
    is_likely_english,
    normalize_subjects,
    parse_description,
)
from src.etl.schema import Book

# HuggingFace cache paths (set after download)
HF_CACHE = Path(
    r"C:\Users\topos\.cache\huggingface\hub"
    r"\datasets--storytracer--openlibrary_dump_2024-04-30"
    r"\snapshots\556a8975f8e41b71da49a36894c13c66b30352b5"
    r"\data\parquet"
)
WORKS_PATH = HF_CACHE / "ol_dump_works_2024-04-30.parquet"
AUTHORS_PATH = HF_CACHE / "ol_dump_authors_2024-04-30.parquet"
OUTPUT_DIR = Path("data/processed")


def build_author_lookup() -> dict[str, str]:
    """Build a key→name lookup from the authors parquet."""
    print("Building author lookup table...")
    t0 = time.time()

    pf = pq.ParquetFile(str(AUTHORS_PATH))
    # Only read key and name columns for memory efficiency
    table = pf.read(columns=["key", "name"])
    keys = table.column("key").to_pylist()
    names = table.column("name").to_pylist()

    lookup = {}
    for k, n in zip(keys, names):
        if k and n:
            lookup[k] = n

    elapsed = time.time() - t0
    print(f"  Loaded {len(lookup):,} authors in {elapsed:.1f}s")
    return lookup


def extract_author_keys(authors_field) -> list[str]:
    """Extract author keys from OL's nested author structure."""
    if authors_field is None:
        return []
    if isinstance(authors_field, np.ndarray):
        authors_field = authors_field.tolist()
    if not isinstance(authors_field, list):
        return []

    keys = []
    for entry in authors_field:
        if isinstance(entry, dict):
            author_ref = entry.get("author", {})
            if isinstance(author_ref, dict):
                key = author_ref.get("key", "")
                if key:
                    keys.append(key)
    return keys


def extract_cover_id(covers_field) -> int | None:
    """Get the first valid cover ID."""
    if covers_field is None:
        return None
    if isinstance(covers_field, np.ndarray):
        covers_field = covers_field.tolist()
    if isinstance(covers_field, list) and covers_field:
        cid = covers_field[0]
        if isinstance(cid, (int, np.integer)) and cid > 0:
            return int(cid)
    return None


def safe_list(val) -> list:
    """Convert numpy arrays or None to plain lists."""
    if val is None:
        return []
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, list):
        return val
    return []


def process_works(
    author_lookup: dict[str, str],
    max_rows: int | None = None,
    tier_filter: int | None = None,
    english_only: bool = True,
) -> list[Book]:
    """Process works parquet into cleaned Book records."""
    print(f"\nProcessing works (max_rows={max_rows}, tier_filter={tier_filter})...")
    t0 = time.time()

    pf = pq.ParquetFile(str(WORKS_PATH))
    total_rows = pf.metadata.num_rows
    print(f"  Total rows in parquet: {total_rows:,}")

    books = []
    stats = {"total_read": 0, "tier1": 0, "tier2": 0, "tier3": 0,
             "skipped_no_title": 0, "skipped_english": 0, "skipped_tier": 0}

    num_groups = pf.metadata.num_row_groups
    if max_rows:
        # Estimate how many row groups we need
        avg_per_group = total_rows / num_groups
        groups_needed = min(num_groups, int(max_rows / avg_per_group) + 2)
    else:
        groups_needed = num_groups

    for group_idx in range(groups_needed):
        table = pf.read_row_group(group_idx)
        df = table.to_pandas()

        for i in range(len(df)):
            if max_rows and stats["total_read"] >= max_rows:
                break

            row = df.iloc[i]
            stats["total_read"] += 1

            # Title is required
            title = row.get("title")
            if not title or not isinstance(title, str) or not title.strip():
                stats["skipped_no_title"] += 1
                continue

            title = title.strip()

            # Parse description
            desc = parse_description(row.get("description"))

            # Normalize subjects
            subjects = normalize_subjects(safe_list(row.get("subjects")))

            # Determine tier
            if desc and subjects:
                tier = 1
            elif subjects:
                tier = 2
            else:
                tier = 3

            stats[f"tier{tier}"] += 1

            # Apply tier filter
            if tier_filter is not None and tier > tier_filter:
                stats["skipped_tier"] += 1
                continue

            # English filter (only for books with descriptions)
            if english_only and desc and not is_likely_english(desc):
                stats["skipped_english"] += 1
                continue

            # Resolve author names
            author_keys = extract_author_keys(row.get("authors"))
            author_names = []
            for key in author_keys:
                name = author_lookup.get(key)
                if name:
                    author_names.append(name)

            # Extract other fields
            year = extract_year(row.get("first_publish_date"))
            cover_id = extract_cover_id(row.get("covers"))
            subject_places = normalize_subjects(safe_list(row.get("subject_places")))
            subject_people = normalize_subjects(safe_list(row.get("subject_people")))
            subject_times = normalize_subjects(safe_list(row.get("subject_times")))

            book = Book(
                work_id=row.get("key", ""),
                title=title,
                authors=author_names,
                description=desc,
                subjects=subjects,
                first_publish_year=year,
                cover_id=cover_id,
                subject_places=subject_places,
                subject_people=subject_people,
                subject_times=subject_times,
                tier=tier,
            )
            books.append(book)

        if max_rows and stats["total_read"] >= max_rows:
            break

        if (group_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {stats['total_read']:,} rows, "
                  f"{len(books):,} books kept ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Read:    {stats['total_read']:,}")
    print(f"  Tier 1:  {stats['tier1']:,} (title + desc + subjects)")
    print(f"  Tier 2:  {stats['tier2']:,} (title + subjects)")
    print(f"  Tier 3:  {stats['tier3']:,} (title only)")
    print(f"  Skipped: {stats['skipped_no_title']:,} no title, "
          f"{stats['skipped_english']:,} non-English, "
          f"{stats['skipped_tier']:,} below tier threshold")
    print(f"  Output:  {len(books):,} books")

    return books


def export_jsonl(books: list[Book], output_path: Path):
    """Export books to newline-delimited JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for book in books:
            f.write(book.model_dump_json() + "\n")
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\nExported {len(books):,} books to {output_path} ({size_mb:.1f} MB)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="OpenLibrary ETL Pipeline")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Max rows to read from works parquet")
    parser.add_argument("--tier", type=int, default=None, choices=[1, 2, 3],
                        help="Only keep books at this tier or above (1=richest)")
    parser.add_argument("--no-english-filter", action="store_true",
                        help="Disable English language filter")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path")
    args = parser.parse_args()

    # Build author lookup
    author_lookup = build_author_lookup()

    # Process works
    books = process_works(
        author_lookup=author_lookup,
        max_rows=args.max_rows,
        tier_filter=args.tier,
        english_only=not args.no_english_filter,
    )

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        tier_label = f"tier{args.tier}" if args.tier else "all"
        rows_label = f"{args.max_rows // 1000}k" if args.max_rows else "full"
        out_path = OUTPUT_DIR / f"books_{tier_label}_{rows_label}.jsonl"

    export_jsonl(books, out_path)

    # Show some samples
    print("\n--- Sample books ---")
    for book in books[:5]:
        print(f"\n  {book.title}")
        print(f"    Authors: {', '.join(book.authors) or 'Unknown'}")
        print(f"    Tier: {book.tier}")
        if book.description:
            print(f"    Desc: {book.description[:120]}...")
        print(f"    Subjects: {', '.join(book.subjects[:5])}")
        print(f"    Embed text: {book.embedding_text()[:150]}...")


if __name__ == "__main__":
    main()
