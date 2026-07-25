"""
Augment tier-2 OpenLibrary books with Goodreads descriptions.

Usage:
    python -m src.etl.augment_goodreads
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = ROOT_DIR / "data" / "processed" / "books_tier1-2_500k.jsonl"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "books_augmented.jsonl"
GOODREADS_DATASET = "booksouls/goodreads-book-descriptions"
GOODREADS_SPLIT = "train"
DEFAULT_MIN_DESCRIPTION_LENGTH = 30
DEFAULT_ASCII_THRESHOLD = 0.80
GOODREADS_TOTAL_ESTIMATE = 1_000_000
BOILERPLATE_PATTERNS = (
    re.compile(r"^no description(?: available| provided)?[.!]*$", re.IGNORECASE),
    re.compile(r"^description not available[.!]*$", re.IGNORECASE),
    re.compile(r"^no synopsis(?: available)?[.!]*$", re.IGNORECASE),
    re.compile(r"^currently unavailable[.!]*$", re.IGNORECASE),
    re.compile(r"^n/?a[.!]*$", re.IGNORECASE),
    re.compile(r"^none[.!]*$", re.IGNORECASE),
)


def collapse_whitespace(text: str) -> str:
    """Normalize internal whitespace to single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_title(title: str) -> str:
    """Unicode-normalize a title, strip punctuation, and collapse whitespace."""
    normalized = unicodedata.normalize("NFKD", title)
    pieces: list[str] = []

    for char in normalized:
        if unicodedata.combining(char):
            continue

        category = unicodedata.category(char)
        if category.startswith(("P", "S")) or char.isspace():
            pieces.append(" ")
        else:
            pieces.append(char.casefold())

    return collapse_whitespace("".join(pieces))


def has_description(record: dict[str, Any]) -> bool:
    """Return True when the record has a non-empty description."""
    description = record.get("description")
    return isinstance(description, str) and bool(description.strip())


def is_tier2_candidate(record: dict[str, Any]) -> bool:
    """Return True for tier-2 books that are eligible for Goodreads augmentation."""
    if has_description(record):
        return False

    tier = record.get("tier")
    if tier == 2:
        return True

    if tier is None:
        subjects = record.get("subjects")
        return isinstance(subjects, list) and bool(subjects)

    return False


def is_ascii_heavy(text: str, threshold: float) -> bool:
    """Heuristic English check based on ASCII character ratio."""
    relevant = [char for char in text if not char.isspace()]
    if not relevant:
        return False
    ascii_count = sum(char.isascii() for char in relevant)
    return (ascii_count / len(relevant)) >= threshold


def is_boilerplate_description(text: str) -> bool:
    """Reject placeholder descriptions."""
    candidate = collapse_whitespace(text).casefold()
    return any(pattern.fullmatch(candidate) for pattern in BOILERPLATE_PATTERNS)


def clean_goodreads_description(
    value: Any,
    *,
    min_length: int,
    ascii_threshold: float,
) -> str | None:
    """Validate and normalize a Goodreads description."""
    if not isinstance(value, str):
        return None

    text = collapse_whitespace(unicodedata.normalize("NFKC", value))
    if len(text) < min_length:
        return None
    if is_boilerplate_description(text):
        return None
    if not is_ascii_heavy(text, ascii_threshold):
        return None
    return text


def choose_better_description(current: str | None, candidate: str) -> str:
    """Prefer longer descriptions when duplicate normalized titles exist."""
    if current is None or len(candidate) > len(current):
        return candidate
    return current


def iter_jsonl(path: Path) -> tuple[int, dict[str, Any]]:
    """Yield line numbers and parsed JSON objects from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: skipping malformed JSON on line {line_number:,} of {path}")


def collect_candidate_titles(input_path: Path) -> tuple[int, set[str]]:
    """Scan the input corpus and collect normalized titles for tier-2 books."""
    candidate_titles: set[str] = set()
    candidate_count = 0

    progress = tqdm(desc="Scanning OpenLibrary input", unit="books")
    for _, record in iter_jsonl(input_path):
        progress.update(1)
        if not is_tier2_candidate(record):
            continue

        title = record.get("title")
        if not isinstance(title, str) or not title.strip():
            continue

        normalized_title = normalize_title(title)
        if not normalized_title:
            continue

        candidate_count += 1
        candidate_titles.add(normalized_title)

    progress.close()
    return candidate_count, candidate_titles


def build_goodreads_index(
    candidate_titles: set[str],
    *,
    dataset_name: str,
    split: str,
    streaming: bool,
    min_length: int,
    ascii_threshold: float,
) -> dict[str, str]:
    """Build a normalized-title → description lookup for matching tier-2 books."""
    if not candidate_titles:
        return {}

    print(
        f"Loading Goodreads dataset ({dataset_name}, split={split}, streaming={streaming})..."
    )
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    matches: dict[str, str] = {}
    progress = tqdm(
        total=GOODREADS_TOTAL_ESTIMATE if streaming else None,
        desc="Scanning Goodreads",
        unit="rows",
    )

    for row in dataset:
        progress.update(1)

        title = row.get("title")
        if not isinstance(title, str) or not title.strip():
            continue

        description = clean_goodreads_description(
            row.get("description"),
            min_length=min_length,
            ascii_threshold=ascii_threshold,
        )
        if description is None:
            continue

        normalized_title = normalize_title(title)
        if normalized_title not in candidate_titles:
            continue

        matches[normalized_title] = choose_better_description(
            matches.get(normalized_title),
            description,
        )

        if len(matches) == len(candidate_titles):
            break

        if progress.n % 50_000 == 0:
            progress.set_postfix_str(f"matched_titles={len(matches):,}")

    progress.close()
    return matches


def write_augmented_corpus(
    input_path: Path,
    output_path: Path,
    title_to_description: dict[str, str],
) -> tuple[int, int]:
    """Write the augmented corpus and return (matched_books, new_tier1_count)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matched_books = 0
    tier1_count = 0

    with output_path.open("w", encoding="utf-8") as handle:
        progress = tqdm(desc="Writing augmented corpus", unit="books")
        for _, record in iter_jsonl(input_path):
            progress.update(1)

            title = record.get("title")
            normalized_title = (
                normalize_title(title) if isinstance(title, str) and title.strip() else ""
            )

            if has_description(record):
                record["description_source"] = "openlibrary"
                tier1_count += 1
            elif is_tier2_candidate(record) and normalized_title in title_to_description:
                record["description"] = title_to_description[normalized_title]
                record["tier"] = 1
                record["description_source"] = "goodreads"
                matched_books += 1
                tier1_count += 1

            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        progress.close()

    return matched_books, tier1_count


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Augment tier-2 OpenLibrary books with Goodreads descriptions."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input JSONL path (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=GOODREADS_DATASET,
        help=f"HuggingFace dataset name (default: {GOODREADS_DATASET})",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=GOODREADS_SPLIT,
        help=f"Dataset split to read (default: {GOODREADS_SPLIT})",
    )
    parser.add_argument(
        "--min-description-length",
        type=int,
        default=DEFAULT_MIN_DESCRIPTION_LENGTH,
        help=f"Minimum Goodreads description length (default: {DEFAULT_MIN_DESCRIPTION_LENGTH})",
    )
    parser.add_argument(
        "--ascii-threshold",
        type=float,
        default=DEFAULT_ASCII_THRESHOLD,
        help=f"Minimum ASCII ratio for accepted descriptions (default: {DEFAULT_ASCII_THRESHOLD})",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming when reading Goodreads.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the augmentation job."""
    args = parse_args()

    candidate_count, candidate_titles = collect_candidate_titles(args.input)
    print(f"Collected {candidate_count:,} tier-2 candidates across {len(candidate_titles):,} titles.")

    title_to_description = build_goodreads_index(
        candidate_titles,
        dataset_name=args.dataset,
        split=args.split,
        streaming=not args.no_streaming,
        min_length=args.min_description_length,
        ascii_threshold=args.ascii_threshold,
    )

    matched_books, new_tier1_count = write_augmented_corpus(
        args.input,
        args.output,
        title_to_description,
    )

    match_rate = (matched_books / candidate_count) if candidate_count else 0.0
    print("\nAugmentation complete.")
    print(f"  Total tier-2 candidates: {candidate_count:,}")
    print(f"  Matches found:           {matched_books:,}")
    print(f"  Match rate:              {match_rate:.2%}")
    print(f"  New tier-1 corpus size:  {new_tier1_count:,}")
    print(f"  Output written to:       {args.output}")


if __name__ == "__main__":
    main()
