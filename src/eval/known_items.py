"""Known-item evaluation dataset.

Each entry maps a natural query to a book that MUST appear in top-k results.
Used as a CI regression gate — if known-item MRR drops below threshold,
something is broken (prefix mismatch, vectorizer swap, bad migration, etc.).

All work_ids verified present in the 26.5K Qdrant corpus (Jul 2025).
"""

KNOWN_ITEMS = [
    {
        "query": "Pride and Prejudice",
        "work_id": "OL24293318W",
        "title": "Pride and prejudice",
    },
    {
        "query": "Romeo and Juliet",
        "work_id": "OL8920022W",
        "title": "Romeo & Juliet",
    },
    {
        "query": "The Great Gatsby",
        "work_id": "OL6934202W",
        "title": "The great Gatsby",
    },
    {
        "query": "To Kill a Mockingbird",
        "work_id": "OL17504684W",
        "title": "To Kill A Mockingbird",
    },
    {
        "query": "Don Quixote",
        "work_id": "OL503666W",
        "title": "Don Quixote",
    },
    {
        "query": "The Odyssey Homer",
        "work_id": "OL5717314W",
        "title": "The Odyssey",
    },
    {
        "query": "The Picture of Dorian Gray",
        "work_id": "OL11317613W",
        "title": "The picture of Dorian Gray",
    },
    {
        "query": "Alice in Wonderland",
        "work_id": "OL32418551W",
        "title": "Alice in wonderland",
    },
    {
        "query": "Treasure Island",
        "work_id": "OL15530609W",
        "title": "Treasure Island",
    },
    {
        "query": "War and Peace Tolstoy",
        "work_id": "OL17463997W",
        "title": "War And Peace",
    },
    {
        "query": "Frankenstein",
        "work_id": "OL19618945W",
        "title": "Frankenstein",
    },
    {
        "query": "Dracula",
        "work_id": "OL24261700W",
        "title": "Dracula",
    },
    {
        "query": "Robinson Crusoe",
        "work_id": "OL17628377W",
        "title": "Robinson Crusoe",
    },
    {
        "query": "Peter Pan",
        "work_id": "OL2889604W",
        "title": "Peter Pan",
    },
    {
        "query": "A Tale of Two Cities",
        "work_id": "OL19908529W",
        "title": "A Tale of two cities",
    },
    {
        "query": "Lord of the Flies",
        "work_id": "OL21110495W",
        "title": "Lord of the Flies",
    },
    {
        "query": "The Wind in the Willows",
        "work_id": "OL20848547W",
        "title": "The wind in the willows",
    },
    {
        "query": "Dune Frank Herbert",
        "work_id": "OL893415W",
        "title": "Dune",
    },
    {
        "query": "The Grapes of Wrath",
        "work_id": "OL15456177W",
        "title": "The grapes of wrath, by John Steinbeck",
    },
    {
        "query": "Charlotte's Web",
        "work_id": "OL27715376W",
        "title": "Charlotte's web by E.B. White",
    },
]
