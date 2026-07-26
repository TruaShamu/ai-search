"""Clean encoding artifacts and Goodreads formatting noise from book descriptions.

Run after augmentation/migration to fix:
  - Goodreads rating block headers (Title / by Author / N ratings / N reviews)
  - Excessive whitespace (\r\n, triple+ newlines)
  - UTF-8 mojibake (double-encoded smart quotes, dashes, accents)

Usage:
    python -m src.etl.clean_descriptions [--input data/index/metadata.jsonl] [--dry-run]
"""

import argparse
import json
import re
from pathlib import Path


def fix_mojibake(text: str) -> str:
    """Fix common UTF-8 double-encoding artifacts."""
    replacements = {
        "\u00e2\u0080\u0099": "\u2019",  # right single quote
        "\u00e2\u0080\u0098": "\u2018",  # left single quote
        "\u00e2\u0080\u009c": "\u201c",  # left double quote
        "\u00e2\u0080\u009d": "\u201d",  # right double quote
        "\u00e2\u0080\u0093": "\u2013",  # en dash
        "\u00e2\u0080\u0094": "\u2014",  # em dash
        "\u00e2\u0080\u00a6": "\u2026",  # ellipsis
        "\u00c2\u00a0": " ",             # non-breaking space
        "\u00c2\u00b7": "\u00b7",        # middle dot
        "\u00c3\u00a9": "\u00e9",        # e-acute
        "\u00c3\u00a1": "\u00e1",        # a-acute
        "\u00c3\u00b1": "\u00f1",        # n-tilde
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def strip_goodreads_header(desc: str) -> str:
    """Remove Goodreads boilerplate header (Title / by Author / ratings / reviews)."""
    match = re.search(r"\d+\s+[Rr]eviews?\s*", desc)
    if match:
        after = desc[match.end():].strip()
        if len(after) > 50:
            return after
    return desc


def normalize_whitespace(desc: str) -> str:
    """Collapse excessive newlines and normalize spacing."""
    desc = desc.replace("\r\n", "\n")
    desc = re.sub(r"\n{3,}", "\n\n", desc)
    desc = re.sub(r"[ \t]+", " ", desc)
    desc = re.sub(r"[ \t]+\n", "\n", desc)
    return desc.strip()


def clean_description(desc: str) -> str:
    """Full cleaning pipeline for a single description."""
    if not desc:
        return desc
    desc = fix_mojibake(desc)
    desc = strip_goodreads_header(desc)
    desc = normalize_whitespace(desc)
    return desc


def main():
    parser = argparse.ArgumentParser(description="Clean book descriptions")
    parser.add_argument("--input", default="data/index/metadata.jsonl", help="Input JSONL path")
    parser.add_argument("--dry-run", action="store_true", help="Don't write changes, just report")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        return

    with input_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    changed = 0
    for i, line in enumerate(lines):
        book = json.loads(line)
        original = book.get("description", "")
        cleaned = clean_description(original)
        if cleaned != original:
            book["description"] = cleaned
            lines[i] = json.dumps(book, ensure_ascii=False) + "\n"
            changed += 1

    print(f"Cleaned {changed} / {len(lines)} descriptions")

    if not args.dry_run:
        with input_path.open("w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"Written to {input_path}")
    else:
        print("(dry-run — no changes written)")


if __name__ == "__main__":
    main()
