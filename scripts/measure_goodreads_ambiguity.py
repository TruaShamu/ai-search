"""Measure how many Goodreads title matches are actually ambiguous.

The ETL joined descriptions by normalized title. Collisions inside the
Goodreads dataset are the corruption source. This counts, for the titles we
actually matched, how many map to more than one distinct description --
i.e. how many of our 13,088 descriptions were a coin flip.
"""

import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from src.etl.augment_goodreads import normalize_title

OUT = Path("data/eval/goodreads_ambiguity.json")


def main() -> None:
    recs = [json.loads(line) for line in open("data/index/metadata.jsonl", encoding="utf-8")]
    matched = {
        normalize_title(r["title"]): r["work_id"]
        for r in recs
        if r.get("description_source") == "goodreads" and r.get("title")
    }
    print(f"corpus titles matched via goodreads: {len(matched):,}")

    ds = load_dataset("booksouls/goodreads-book-descriptions", split="train", streaming=True)

    seen: dict[str, set[str]] = defaultdict(set)
    rows = 0
    for row in ds:
        rows += 1
        t = row.get("title")
        d = row.get("description")
        if not isinstance(t, str) or not isinstance(d, str):
            continue
        nt = normalize_title(t)
        if nt in matched:
            seen[nt].add(d.strip()[:300])
        if rows % 100_000 == 0:
            print(f"  {rows:,} rows scanned", flush=True)

    print(f"\nscanned {rows:,} goodreads rows")

    unique = sum(1 for t in matched if len(seen.get(t, ())) == 1)
    ambiguous = sum(1 for t in matched if len(seen.get(t, ())) > 1)
    missing = sum(1 for t in matched if not seen.get(t))

    print(f"\nof {len(matched):,} matched titles:")
    print(f"  exactly ONE goodreads description  : {unique:,}  ({unique/len(matched):.1%})  <- safe")
    print(f"  MULTIPLE distinct descriptions     : {ambiguous:,}  ({ambiguous/len(matched):.1%})  <- coin flip")
    print(f"  not found on rescan                : {missing:,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "matched_titles": len(matched),
                "unique": unique,
                "ambiguous": ambiguous,
                "missing": missing,
                "ambiguous_titles": sorted(t for t in matched if len(seen.get(t, ())) > 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
