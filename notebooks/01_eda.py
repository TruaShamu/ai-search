"""
OpenLibrary EDA - Phase 0
Reads the works parquet directly from HuggingFace Hub to analyze
data quality, field coverage, and text characteristics.
"""

import re
import sys
from collections import Counter
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
import pandas as pd

DATASET_REPO = "storytracer/openlibrary_dump_2024-04-30"
PARQUET_FILE = "data/parquet/ol_dump_works_2024-04-30.parquet"
SAMPLE_SIZE = 50_000

# ──────────────────────────────────────────────
# 1. Download & load works parquet
# ──────────────────────────────────────────────
print("=== Downloading works parquet from HuggingFace ===\n")
local_path = hf_hub_download(
    repo_id=DATASET_REPO,
    filename=PARQUET_FILE,
    repo_type="dataset",
)
print(f"Downloaded to: {local_path}\n")

print(f"=== Reading parquet (sampling {SAMPLE_SIZE:,} rows) ===\n")
pf = pq.ParquetFile(local_path)
print(f"Total row groups: {pf.metadata.num_row_groups}")
print(f"Total rows: {pf.metadata.num_rows:,}")
print(f"Schema:\n{pf.schema_arrow}\n")

# Read first N rows
table = pf.read_row_groups(range(min(pf.metadata.num_row_groups, 10)))
df = table.to_pandas()
if len(df) > SAMPLE_SIZE:
    df = df.head(SAMPLE_SIZE)

print(f"Loaded {len(df):,} rows for analysis\n")
print(f"Columns: {list(df.columns)}\n")

# ──────────────────────────────────────────────
# 2. Field coverage
# ──────────────────────────────────────────────
print("=" * 60)
print("FIELD COVERAGE")
print("=" * 60)

def non_null_count(series):
    count = 0
    for val in series:
        if val is None:
            continue
        if isinstance(val, (list, dict)) and len(val) == 0:
            continue
        if isinstance(val, str) and val.strip() == "":
            continue
        count += 1
    return count

print(f"\n{'Field':<25} {'Non-empty':>10} {'Coverage':>10}")
print("-" * 47)
for col in df.columns:
    count = non_null_count(df[col])
    pct = count / len(df) * 100
    print(f"{col:<25} {count:>10,} {pct:>9.1f}%")

# ──────────────────────────────────────────────
# 3. Description analysis
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("DESCRIPTION ANALYSIS")
print("=" * 60)

desc_texts = []
for val in df["description"]:
    if val is None:
        continue
    if isinstance(val, str) and val.strip():
        desc_texts.append(val.strip())
    elif isinstance(val, dict):
        v = val.get("value", "")
        if v:
            desc_texts.append(v)

has_desc = len(desc_texts)
print(f"\nBooks with descriptions: {has_desc:,} / {len(df):,} ({has_desc/len(df)*100:.1f}%)")

if desc_texts:
    lengths = [len(t) for t in desc_texts]
    ds = pd.Series(lengths)
    print(f"\nDescription length stats:")
    print(f"  Min:    {ds.min():>8,} chars")
    print(f"  25th:   {int(ds.quantile(0.25)):>8,} chars")
    print(f"  Median: {int(ds.median()):>8,} chars")
    print(f"  Mean:   {int(ds.mean()):>8,} chars")
    print(f"  75th:   {int(ds.quantile(0.75)):>8,} chars")
    print(f"  95th:   {int(ds.quantile(0.95)):>8,} chars")
    print(f"  Max:    {ds.max():>8,} chars")

    print(f"\n--- Sample descriptions ---")
    for i, text in enumerate(desc_texts[:5]):
        preview = text[:200].replace("\n", " ").replace("\r", "")
        print(f"  [{i+1}] ({len(text)} chars) {preview}")
        print()

# ──────────────────────────────────────────────
# 4. First sentence analysis
# ──────────────────────────────────────────────
print("=" * 60)
print("FIRST SENTENCE ANALYSIS")
print("=" * 60)

if "first_sentence" in df.columns:
    fs_texts = []
    for val in df["first_sentence"]:
        if val is None:
            continue
        if isinstance(val, str) and val.strip():
            fs_texts.append(val.strip())
        elif isinstance(val, dict):
            v = val.get("value", "")
            if v:
                fs_texts.append(v)

    has_fs = len(fs_texts)
    print(f"\nBooks with first_sentence: {has_fs:,} / {len(df):,} ({has_fs/len(df)*100:.1f}%)")

    has_desc_set = set()
    has_fs_set = set()
    for i in range(len(df)):
        d = df.iloc[i]["description"]
        if d and ((isinstance(d, str) and d.strip()) or (isinstance(d, dict) and d.get("value", "").strip())):
            has_desc_set.add(i)
        f = df.iloc[i].get("first_sentence")
        if f and ((isinstance(f, str) and f.strip()) or (isinstance(f, dict) and f.get("value", "").strip())):
            has_fs_set.add(i)

    fs_only = has_fs_set - has_desc_set
    print(f"Books with first_sentence BUT NO description: {len(fs_only):,}")

    if fs_texts:
        print(f"\n--- Sample first sentences ---")
        for i, text in enumerate(fs_texts[:5]):
            print(f"  [{i+1}] {text[:200]}")
else:
    print("\nfirst_sentence column not present in this parquet file")

# ──────────────────────────────────────────────
# 5. Title analysis
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("TITLE ANALYSIS")
print("=" * 60)

titles = [t for t in df["title"] if t and isinstance(t, str)]
if titles:
    title_lengths = pd.Series([len(t) for t in titles])
    print(f"\nTitle count: {len(titles):,}")
    print(f"  Min:    {title_lengths.min():>6} chars")
    print(f"  Median: {int(title_lengths.median()):>6} chars")
    print(f"  Mean:   {int(title_lengths.mean()):>6} chars")
    print(f"  Max:    {title_lengths.max():>6} chars")

    title_counter = Counter(titles)
    dupes = [(t, c) for t, c in title_counter.most_common(15) if c > 1]
    if dupes:
        print(f"\nMost duplicated titles:")
        for title, count in dupes[:10]:
            print(f"  {count:>5}x  {title[:80]}")

# ──────────────────────────────────────────────
# 6. Subject analysis
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUBJECT ANALYSIS")
print("=" * 60)

all_subjects = []
books_with_subjects = 0
for subs in df["subjects"]:
    if subs and isinstance(subs, list) and len(subs) > 0:
        books_with_subjects += 1
        all_subjects.extend(subs)

subject_counter = Counter(all_subjects)
print(f"\nBooks with subjects: {books_with_subjects:,} / {len(df):,} ({books_with_subjects/len(df)*100:.1f}%)")
print(f"Unique subjects: {len(subject_counter):,}")
print(f"Avg subjects per book: {len(all_subjects)/max(books_with_subjects,1):.1f}")

print(f"\nTop 25 subjects:")
for sub, count in subject_counter.most_common(25):
    print(f"  {count:>6,}  {sub}")

print(f"\nSubject messiness - case variants of 'Fiction':")
fiction_variants = [(s, c) for s, c in subject_counter.items()
                    if s.lower().strip() == "fiction"]
for s, c in sorted(fiction_variants, key=lambda x: -x[1]):
    print(f"  {c:>6,}  '{s}'")

# ──────────────────────────────────────────────
# 7. Author structure
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("AUTHOR ANALYSIS")
print("=" * 60)

books_with_authors = 0
author_counts_list = []
sample_authors = None
for authors in df["authors"]:
    if authors and isinstance(authors, list) and len(authors) > 0:
        books_with_authors += 1
        author_counts_list.append(len(authors))
        if sample_authors is None:
            sample_authors = authors[:3]

print(f"\nBooks with author refs: {books_with_authors:,} / {len(df):,} ({books_with_authors/len(df)*100:.1f}%)")
if author_counts_list:
    acs = pd.Series(author_counts_list)
    print(f"  Min: {acs.min()}, Median: {int(acs.median())}, Max: {acs.max()}")

if sample_authors:
    print(f"\nSample author structure:")
    for a in sample_authors:
        print(f"  {a}")
    print("  -> Authors are keys (e.g. /authors/OL123A), need join to get names")

# ──────────────────────────────────────────────
# 8. Publication date analysis
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("PUBLICATION DATE ANALYSIS")
print("=" * 60)

years = []
date_formats = Counter()
for fpd in df["first_publish_date"]:
    if fpd is None or (isinstance(fpd, str) and fpd.strip() == ""):
        date_formats["missing"] += 1
        continue
    fpd_str = str(fpd)
    match = re.search(r"(\d{4})", fpd_str)
    if match:
        year = int(match.group(1))
        if 1000 <= year <= 2030:
            years.append(year)
            date_formats["has_year"] += 1
        else:
            date_formats["invalid_year"] += 1
    else:
        date_formats["no_year_found"] += 1

print(f"\nDate field distribution:")
for fmt, count in date_formats.most_common():
    pct = count / len(df) * 100
    print(f"  {fmt:<20} {count:>8,}  ({pct:.1f}%)")

if years:
    ys = pd.Series(years)
    print(f"\nPublication year stats (n={len(years):,}):")
    print(f"  Min: {ys.min()}, Median: {int(ys.median())}, Max: {ys.max()}")

    decade_counter = Counter((y // 10) * 10 for y in years)
    print(f"\nBooks by decade (top 10):")
    for decade, count in sorted(decade_counter.items(), key=lambda x: -x[1])[:10]:
        bar = chr(9608) * max(1, count * 40 // max(decade_counter.values()))
        print(f"  {decade}s: {count:>6,}  {bar}")

# ──────────────────────────────────────────────
# 9. Covers analysis
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("COVER IMAGES")
print("=" * 60)

books_with_covers = sum(1 for c in df["covers"] if c and isinstance(c, list) and len(c) > 0)
print(f"\nBooks with cover IDs: {books_with_covers:,} / {len(df):,} ({books_with_covers/len(df)*100:.1f}%)")
print(f"  Cover URL pattern: https://covers.openlibrary.org/b/id/{{cover_id}}-M.jpg")

# ──────────────────────────────────────────────
# 10. Embedding input preview
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("EMBEDDING INPUT PREVIEW")
print("=" * 60)

preview_count = 0
for idx in range(len(df)):
    row = df.iloc[idx]
    title = row.get("title")
    desc = row.get("description")
    if isinstance(desc, dict):
        desc = desc.get("value", "")
    if not desc or not title:
        continue

    subjects = row.get("subjects") or []
    authors = row.get("authors") or []
    author_keys = []
    for a in authors[:3]:
        if isinstance(a, dict):
            ak = a.get("author", {})
            if isinstance(ak, dict):
                author_keys.append(ak.get("key", ""))
            else:
                author_keys.append(str(ak))
    author_str = ", ".join(author_keys) if author_keys else "Unknown"

    subject_str = ", ".join(subjects[:5]) if subjects else ""
    desc_str = desc[:300] if isinstance(desc, str) else ""
    embed_text = f"{title} by {author_str}. {desc_str}. {subject_str}"

    print(f"\n  [{preview_count+1}] ({len(embed_text)} chars)")
    print(f"      {embed_text[:300]}")

    preview_count += 1
    if preview_count >= 5:
        break

# ──────────────────────────────────────────────
# 11. Usable records estimate
# ──────────────────────────────────────────────
print("\n\n" + "=" * 60)
print("FILTERING ANALYSIS - What's usable for search?")
print("=" * 60)

has_title_mask = df["title"].apply(lambda x: bool(x and isinstance(x, str) and x.strip()))

has_desc_mask = df["description"].apply(lambda x: bool(
    (isinstance(x, str) and x.strip()) or
    (isinstance(x, dict) and x.get("value", "").strip())
))

has_subjects_mask = df["subjects"].apply(lambda x: bool(x and isinstance(x, list) and len(x) > 0))

has_any_text = has_title_mask & (has_desc_mask | has_subjects_mask)
has_rich = has_title_mask & has_desc_mask & has_subjects_mask

total_rows = pf.metadata.num_rows

print(f"\nIn sample ({len(df):,} rows):")
print(f"  Has title:                    {has_title_mask.sum():>8,} ({has_title_mask.sum()/len(df)*100:.1f}%)")
print(f"  Has title + desc:             {(has_title_mask & has_desc_mask).sum():>8,} ({(has_title_mask & has_desc_mask).sum()/len(df)*100:.1f}%)")
print(f"  Has title + subjects:         {(has_title_mask & has_subjects_mask).sum():>8,} ({(has_title_mask & has_subjects_mask).sum()/len(df)*100:.1f}%)")
print(f"  Has title + (desc OR subj):   {has_any_text.sum():>8,} ({has_any_text.sum()/len(df)*100:.1f}%)")
print(f"  Has title + desc + subjects:  {has_rich.sum():>8,} ({has_rich.sum()/len(df)*100:.1f}%)")

print(f"\nExtrapolated to full dataset ({total_rows:,} total works):")
for label, mask in [
    ("title + desc", has_title_mask & has_desc_mask),
    ("title + (desc OR subj)", has_any_text),
    ("title + desc + subj (richest)", has_rich),
]:
    pct = mask.sum() / len(df)
    est = int(pct * total_rows)
    print(f"  {label:<35} ~{est:>10,}")

# ──────────────────────────────────────────────
# 12. Summary
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 60)

desc_pct = has_desc_mask.sum() / len(df) * 100
subj_pct = has_subjects_mask.sum() / len(df) * 100
auth_pct = books_with_authors / len(df) * 100

print(f"""
Dataset: {DATASET_REPO} (works parquet)
Total works in dump: {total_rows:,}
Sample analyzed: {len(df):,}

Key findings:
  1. ~{desc_pct:.0f}% of works have descriptions - need enrichment for the rest
  2. ~{subj_pct:.0f}% have subjects, but they're messy (case variants, near-dupes)
  3. Authors are stored as keys - need join with authors dump to get names
  4. Dates are inconsistent but extractable with regex
  5. first_sentence field can supplement missing descriptions

Recommendations:
  - Start Phase 0 prototype with 'title + desc + subjects' subset
  - Build subject normalization early (lowercase, merge variants)
  - Join authors dump to resolve names - critical for search quality
  - Consider LLM-generating descriptions for title-only works in Phase 2+
  - Embedding input: "{{title}} by {{author_name}}. {{description}}. {{subjects}}"
""")
