"""Check enrichment field coverage in exported data."""
import json

has_year = 0
has_places = 0
has_people = 0
has_times = 0
total = 0

with open("data/processed/books_tier1-2_500k.jsonl", encoding="utf-8") as f:
    for line in f:
        book = json.loads(line)
        total += 1
        if book.get("first_publish_year"):
            has_year += 1
        if book.get("subject_places") and len(book["subject_places"]) > 0:
            has_places += 1
        if book.get("subject_people") and len(book["subject_people"]) > 0:
            has_people += 1
        if book.get("subject_times") and len(book["subject_times"]) > 0:
            has_times += 1

print(f"Total books: {total:,}")
print(f"first_publish_year:  {has_year:>8,} ({has_year/total*100:.1f}%)")
print(f"subject_places:      {has_places:>8,} ({has_places/total*100:.1f}%)")
print(f"subject_people:      {has_people:>8,} ({has_people/total*100:.1f}%)")
print(f"subject_times:       {has_times:>8,} ({has_times/total*100:.1f}%)")

# Richest examples
print("\n--- Richest examples (all fields present) ---")
count = 0
with open("data/processed/books_tier1-2_500k.jsonl", encoding="utf-8") as f:
    for line in f:
        b = json.loads(line)
        if (b.get("first_publish_year") and b.get("subject_places")
                and b.get("subject_people") and b.get("description")):
            a = ", ".join(b["authors"]) or "?"
            subs = ", ".join(b["subjects"][:5])
            places = ", ".join(b["subject_places"][:3])
            people = ", ".join(b["subject_people"][:3])
            times = ", ".join(b.get("subject_times", [])[:3])
            desc = b["description"][:120]
            print(f"\n  Title:    {b['title']}")
            print(f"  Author:   {a}")
            print(f"  Year:     {b['first_publish_year']}")
            print(f"  Subjects: {subs}")
            print(f"  Places:   {places}")
            print(f"  People:   {people}")
            print(f"  Times:    {times}")
            print(f"  Desc:     {desc}...")
            count += 1
            if count >= 5:
                break

# Also show what a Tier 2 book looks like with year
print("\n--- Tier 2 examples with year ---")
count = 0
with open("data/processed/books_tier1-2_500k.jsonl", encoding="utf-8") as f:
    for line in f:
        b = json.loads(line)
        if b["tier"] == 2 and b.get("first_publish_year") and b.get("subjects"):
            a = ", ".join(b["authors"]) or "?"
            subs = ", ".join(b["subjects"][:5])
            print(f"\n  Title:    {b['title']}")
            print(f"  Author:   {a}")
            print(f"  Year:     {b['first_publish_year']}")
            print(f"  Subjects: {subs}")
            count += 1
            if count >= 3:
                break
