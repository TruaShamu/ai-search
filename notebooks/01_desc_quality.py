"""Check description quality — are they actually useful for embeddings?"""

import json
import re
from collections import Counter
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

LOCAL_PATH = r"C:\Users\topos\.cache\huggingface\hub\datasets--storytracer--openlibrary_dump_2024-04-30\snapshots\556a8975f8e41b71da49a36894c13c66b30352b5\data\parquet\ol_dump_works_2024-04-30.parquet"

pf = pq.ParquetFile(LOCAL_PATH)
table = pf.read_row_groups(range(min(pf.metadata.num_row_groups, 10)))
df = table.to_pandas().head(50_000)

def extract_desc(val):
    if val is None or (isinstance(val, str) and val.strip() == ""):
        return None
    if isinstance(val, str):
        val = val.strip()
        if val.startswith("{"):
            try:
                parsed = json.loads(val)
                txt = parsed.get("value", "").strip()
                return txt if txt else None
            except Exception:
                return val
        return val
    return None

descs = [(i, extract_desc(v)) for i, v in enumerate(df["description"]) if extract_desc(v)]

print(f"Total descriptions: {len(descs):,}\n")

# ── Classify description quality ──
categories = Counter()
examples = {}

for idx, text in descs:
    length = len(text)

    # Physical description (page counts, dimensions)
    if re.match(r"^\d+\s*(p\.|pages|v\.|vol)", text, re.IGNORECASE) or "cm" in text[:30]:
        cat = "physical_only"
    # Very short / useless
    elif length < 20:
        cat = "too_short"
    # Non-English (rough heuristic)
    elif length > 30 and not any(w in text.lower() for w in ["the", "a ", "is ", "and", "of ", "in ", "for", "this", "an "]):
        cat = "likely_non_english"
    # Contains meaningful narrative
    elif length >= 50 and any(w in text.lower() for w in ["story", "novel", "account", "explores", "tells", "follows", "about", "journey", "life", "world", "history"]):
        cat = "good_narrative"
    # Decent length, probably useful
    elif length >= 100:
        cat = "decent_length"
    elif length >= 30:
        cat = "short_but_ok"
    else:
        cat = "unclear"

    categories[cat] += 1
    if cat not in examples:
        examples[cat] = []
    if len(examples[cat]) < 3:
        title = df.iloc[idx]["title"] or "?"
        examples[cat].append((title, text[:200]))

print("=" * 60)
print("DESCRIPTION QUALITY BREAKDOWN")
print("=" * 60)
print()

quality_order = ["good_narrative", "decent_length", "short_but_ok", "likely_non_english", "physical_only", "too_short", "unclear"]
for cat in quality_order:
    count = categories.get(cat, 0)
    pct = count / len(descs) * 100
    emoji = {
        "good_narrative": "[GOOD]",
        "decent_length": "[OK]  ",
        "short_but_ok": "[WEAK]",
        "likely_non_english": "[SKIP]",
        "physical_only": "[JUNK]",
        "too_short": "[JUNK]",
        "unclear": "[????]",
    }.get(cat, "     ")
    print(f"  {emoji} {cat:<25} {count:>5,}  ({pct:.1f}%)")

print("\n" + "=" * 60)
print("EXAMPLES BY CATEGORY")
print("=" * 60)

for cat in quality_order:
    if cat in examples:
        print(f"\n--- {cat} ---")
        for title, text in examples[cat]:
            print(f"  [{title[:50]}]")
            print(f"    {text[:180]}")
            print()

# ── Language distribution of descriptions ──
print("=" * 60)
print("LANGUAGE HEURISTIC (of descriptions)")
print("=" * 60)

lang_counts = Counter()
EN_WORDS = {"the", "a", "an", "is", "are", "was", "were", "and", "of", "in", "to", "for", "this", "that", "with", "from", "by", "on", "it", "as", "at", "be", "or", "not", "but", "have", "has", "had", "her", "his"}

for _, text in descs:
    words = set(text.lower().split()[:20])
    en_overlap = len(words & EN_WORDS)
    if en_overlap >= 3:
        lang_counts["english"] += 1
    elif en_overlap >= 1:
        lang_counts["maybe_english"] += 1
    else:
        lang_counts["non_english"] += 1

print()
for lang, count in lang_counts.most_common():
    pct = count / len(descs) * 100
    print(f"  {lang:<20} {count:>5,}  ({pct:.1f}%)")

# ── What about books WITHOUT descriptions but WITH subjects? ──
print("\n" + "=" * 60)
print("SUBJECTS AS FALLBACK (for books without descriptions)")
print("=" * 60)

no_desc_with_subjects = 0
subject_only_examples = []
for i in range(len(df)):
    desc = extract_desc(df.iloc[i]["description"])
    subs = df.iloc[i]["subjects"]
    if isinstance(subs, np.ndarray):
        subs = subs.tolist()
    if desc is None and isinstance(subs, list) and len(subs) >= 2:
        no_desc_with_subjects += 1
        if len(subject_only_examples) < 8:
            title = df.iloc[i]["title"] or "?"
            subject_only_examples.append((title, subs[:8]))

print(f"\nBooks with NO desc but HAVE subjects: {no_desc_with_subjects:,}")
print(f"\nSamples (title + subjects as embedding fallback):")
for title, subs in subject_only_examples:
    embed_fallback = f"{title}. {', '.join(subs)}"
    print(f"  {embed_fallback[:150]}")

# ── Final verdict ──
good = categories.get("good_narrative", 0) + categories.get("decent_length", 0)
ok = categories.get("short_but_ok", 0)
junk = categories.get("physical_only", 0) + categories.get("too_short", 0)
non_en = categories.get("likely_non_english", 0)

print("\n" + "=" * 60)
print("VERDICT")
print("=" * 60)
print(f"""
Of {len(descs):,} descriptions in sample:
  GOOD for embeddings:    {good:>5,} ({good/len(descs)*100:.1f}%) — narrative, 100+ chars
  WEAK but usable:        {ok:>5,} ({ok/len(descs)*100:.1f}%) — short but has content
  NON-ENGLISH:            {non_en:>5,} ({non_en/len(descs)*100:.1f}%) — need language filter
  JUNK (physical/short):  {junk:>5,} ({junk/len(descs)*100:.1f}%) — page counts, too short

Extrapolated to full dump (34.6M works):
  Usable English descriptions:  ~{int((good + ok - non_en * 0.5) / len(descs) * 34_666_230 * len(descs) / 50_000):,}
  Books with subjects fallback: ~{int(no_desc_with_subjects / 50_000 * 34_666_230):,}

RECOMMENDATION:
  - DON'T limit to only description-having books (too few)
  - Use a TIERED embedding strategy:
    Tier 1: title + desc + subjects  (~1.6M works)  — best quality
    Tier 2: title + subjects         (~21M works)   — good enough
    Tier 3: title only               (~11M works)   — minimal, skip in v1
  - Filter out non-English descriptions
  - Filter out junk descriptions (page counts, < 20 chars)
  - Start Phase 0 with Tier 1 (richest ~50K records)
""")
