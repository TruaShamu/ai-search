"""EDA Part 2 — subjects, authors, dates, filtering analysis."""

import re
import json
from collections import Counter
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

LOCAL_PATH = r"C:\Users\topos\.cache\huggingface\hub\datasets--storytracer--openlibrary_dump_2024-04-30\snapshots\556a8975f8e41b71da49a36894c13c66b30352b5\data\parquet\ol_dump_works_2024-04-30.parquet"

pf = pq.ParquetFile(LOCAL_PATH)
table = pf.read_row_groups(range(min(pf.metadata.num_row_groups, 10)))
df = table.to_pandas().head(50_000)
total_rows = pf.metadata.num_rows
print(f"Loaded {len(df):,} rows, total in dump: {total_rows:,}\n")


def extract_desc(val):
    """Parse description — handles JSON-encoded and plain string formats."""
    if val is None or (isinstance(val, str) and val.strip() == ""):
        return None
    if isinstance(val, str):
        val = val.strip()
        if val.startswith("{"):
            try:
                parsed = json.loads(val)
                txt = parsed.get("value", "").strip()
                return txt if len(txt) > 10 else None
            except Exception:
                return val if len(val) > 10 else None
        return val if len(val) > 10 else None
    return None


# ── Descriptions (re-analyze with JSON parsing) ──
desc_texts = [d for d in (extract_desc(v) for v in df["description"]) if d]
print("=" * 60)
print("DESCRIPTIONS (parsed from JSON)")
print("=" * 60)
print(f"Books with descriptions: {len(desc_texts):,} / {len(df):,} ({len(desc_texts)/len(df)*100:.1f}%)")
if desc_texts:
    ds = pd.Series([len(t) for t in desc_texts])
    print(f"  Min: {ds.min():,}, Median: {int(ds.median()):,}, Mean: {int(ds.mean()):,}, Max: {ds.max():,}")
    print("\nSample parsed descriptions:")
    for i, text in enumerate(desc_texts[:5]):
        print(f"  [{i+1}] ({len(text)} chars) {text[:180]}")
        print()

# ── Subjects ──
print("=" * 60)
print("SUBJECT ANALYSIS")
print("=" * 60)

all_subjects = []
books_with_subjects = 0
for subs in df["subjects"]:
    try:
        if isinstance(subs, np.ndarray):
            subs = subs.tolist()
        if isinstance(subs, list) and len(subs) > 0:
            books_with_subjects += 1
            all_subjects.extend(subs)
    except Exception:
        pass

subject_counter = Counter(all_subjects)
print(f"\nBooks with subjects: {books_with_subjects:,} / {len(df):,} ({books_with_subjects/len(df)*100:.1f}%)")
print(f"Unique subjects: {len(subject_counter):,}")
print(f"Avg subjects per book: {len(all_subjects)/max(books_with_subjects,1):.1f}")

print("\nTop 25 subjects:")
for sub, count in subject_counter.most_common(25):
    print(f"  {count:>6,}  {sub}")

print("\nCase variants of 'Fiction':")
fiction_variants = [
    (s, c) for s, c in subject_counter.items()
    if isinstance(s, str) and s.lower().strip() == "fiction"
]
for s, c in sorted(fiction_variants, key=lambda x: -x[1]):
    print(f"  {c:>6,}  \"{s}\"")

# ── Authors ──
print("\n" + "=" * 60)
print("AUTHOR ANALYSIS")
print("=" * 60)

books_with_authors = 0
sample_author = None
for authors in df["authors"]:
    try:
        if isinstance(authors, np.ndarray):
            authors = authors.tolist()
        if isinstance(authors, list) and len(authors) > 0:
            books_with_authors += 1
            if sample_author is None:
                sample_author = authors[0]
    except Exception:
        pass

print(f"\nBooks with author refs: {books_with_authors:,} / {len(df):,} ({books_with_authors/len(df)*100:.1f}%)")
if sample_author:
    print(f"Sample author entry: {sample_author}")
    print("  -> Authors are keys like /authors/OL123A, need join with authors dump")

# ── Dates ──
print("\n" + "=" * 60)
print("PUBLICATION DATE ANALYSIS")
print("=" * 60)

years = []
date_formats = Counter()
for fpd in df["first_publish_date"]:
    if not fpd or not isinstance(fpd, str) or not fpd.strip():
        date_formats["missing"] += 1
        continue
    match = re.search(r"(\d{4})", fpd)
    if match:
        year = int(match.group(1))
        if 1000 <= year <= 2030:
            years.append(year)
            date_formats["has_year"] += 1
        else:
            date_formats["invalid"] += 1
    else:
        date_formats["no_year"] += 1

print(f"\nDate coverage:")
for fmt, count in date_formats.most_common():
    print(f"  {fmt:<15} {count:>8,}  ({count/len(df)*100:.1f}%)")

if years:
    ys = pd.Series(years)
    print(f"\nYear stats (n={len(years):,}): min={ys.min()}, median={int(ys.median())}, max={ys.max()}")

    decade_counter = Counter((y // 10) * 10 for y in years)
    print("\nBooks by decade (top 10):")
    max_count = max(decade_counter.values())
    for decade, count in sorted(decade_counter.items(), key=lambda x: -x[1])[:10]:
        bar = chr(9608) * max(1, count * 40 // max_count)
        print(f"  {decade}s: {count:>6,}  {bar}")

# ── Covers ──
print("\n" + "=" * 60)
print("COVERS")
print("=" * 60)

books_with_covers = 0
for c in df["covers"]:
    try:
        if isinstance(c, np.ndarray):
            c = c.tolist()
        if isinstance(c, list) and len(c) > 0:
            books_with_covers += 1
    except Exception:
        pass

print(f"\nBooks with covers: {books_with_covers:,} / {len(df):,} ({books_with_covers/len(df)*100:.1f}%)")

# ── Filtering / Usable Records ──
print("\n" + "=" * 60)
print("FILTERING — What's usable for search?")
print("=" * 60)

has_title = df["title"].apply(lambda x: bool(x and isinstance(x, str) and x.strip()))

has_desc = df["description"].apply(lambda x: extract_desc(x) is not None)

def has_subjects(x):
    try:
        if isinstance(x, np.ndarray):
            return len(x) > 0
        if isinstance(x, list):
            return len(x) > 0
    except Exception:
        pass
    return False

has_subj = df["subjects"].apply(has_subjects)
has_any = has_title & (has_desc | has_subj)
has_rich = has_title & has_desc & has_subj

print(f"\nIn sample ({len(df):,} rows):")
print(f"  title:                    {has_title.sum():>8,} ({has_title.sum()/len(df)*100:.1f}%)")
print(f"  title + desc:             {(has_title & has_desc).sum():>8,} ({(has_title & has_desc).sum()/len(df)*100:.1f}%)")
print(f"  title + subjects:         {(has_title & has_subj).sum():>8,} ({(has_title & has_subj).sum()/len(df)*100:.1f}%)")
print(f"  title + (desc OR subj):   {has_any.sum():>8,} ({has_any.sum()/len(df)*100:.1f}%)")
print(f"  title + desc + subjects:  {has_rich.sum():>8,} ({has_rich.sum()/len(df)*100:.1f}%)")

print(f"\nExtrapolated to full dump ({total_rows:,} works):")
for label, mask in [
    ("title + desc", has_title & has_desc),
    ("title + (desc OR subj)", has_any),
    ("title + desc + subj (richest)", has_rich),
]:
    est = int(mask.sum() / len(df) * total_rows)
    print(f"  {label:<35} ~{est:>10,}")

# ── Summary ──
print("\n" + "=" * 60)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 60)

desc_pct = len(desc_texts) / len(df) * 100
subj_pct = books_with_subjects / len(df) * 100
auth_pct = books_with_authors / len(df) * 100
date_pct = len(years) / len(df) * 100
cover_pct = books_with_covers / len(df) * 100
rich_est = int(has_rich.sum() / len(df) * total_rows)

print(f"""
DATASET: storytracer/openlibrary_dump_2024-04-30 (works)
TOTAL WORKS IN DUMP: {total_rows:,}

FIELD COVERAGE:
  Titles:       ~100%    (nearly all works have a title)
  Descriptions: ~{desc_pct:.0f}%    (LOW — biggest gap for embedding quality)
  Subjects:     ~{subj_pct:.0f}%    (decent, but messy — needs normalization)
  Authors:      ~{auth_pct:.0f}%    (keys only — need join for names)
  Dates:        ~{date_pct:.0f}%    (extractable with regex)
  Covers:       ~{cover_pct:.0f}%

DATA QUALITY ISSUES:
  1. Descriptions are JSON-encoded: {{"type":"/type/text","value":"..."}} — must parse
  2. Only ~{desc_pct:.0f}% have descriptions — limits embedding quality
  3. Subject tags have case variants ("Fiction" vs "fiction" vs "FICTION")
  4. Duplicate titles exist (e.g., "Report" appears 16x)
  5. Author entries are key refs, not names

USABLE RECORDS FOR SEARCH:
  Richest (title + desc + subjects):    ~{rich_est:,} works
  Good enough (title + desc OR subj):   ~{int(has_any.sum() / len(df) * total_rows):,} works

PHASE 0 PROTOTYPE RECOMMENDATIONS:
  1. Filter to works with title + description + subjects (~{rich_est:,})
  2. Parse descriptions from JSON, truncate to 512 chars
  3. Normalize subjects (lowercase, strip whitespace)
  4. Join with authors parquet to get readable names
  5. Start with first 50K rich records for local prototype
  6. Embedding text: "{{title}} by {{author}}. {{description}}. {{subjects}}"
""")
