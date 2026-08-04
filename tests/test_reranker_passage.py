"""Tests for reranker passage building and subject cleaning."""


from src.reranker.onnx_reranker import OnnxReranker
from src.reranker.model import CrossEncoderReranker
from src.reranker.passage import clean_subjects


DUNE_SUBJECTS = [
    "Dune (Imaginary Place)", "Fiction", "Fiction, Science Fiction, General",
    "Dune (Imaginary Place), Fiction", "New York Times Reviewed", "Science Fiction",
    "Science-Fiction", "American Literature", "Nyt:Mass-Market-Monthly=2021-11-07",
    "New York Times Bestseller", "Award:Nebula_Award=Novel",
    "Nyt:Trade-Fiction-Paperback=2021-11-07", "Hugo Award Winner",
    "Award:Hugo_Award=1966", "Award:Hugo_Award=Novel",
]


# ---------------------------------------------------------------------------
# _clean_subjects
# ---------------------------------------------------------------------------

class TestCleanSubjects:

    def test_dune_subjects(self):
        result = clean_subjects(DUNE_SUBJECTS)

        # Machine tokens removed
        for s in result:
            assert "Nyt:" not in s, f"Machine token leaked: {s}"
            assert "Award:" not in s, f"Machine token leaked: {s}"
            assert "=" not in s, f"Token with '=' leaked: {s}"

        # Redundant "Fiction" dropped (substring of "Science Fiction")
        assert "Fiction" not in result

        # "Science Fiction" kept
        assert "Science Fiction" in result

        # "Science-Fiction" deduped against "Science Fiction"
        assert "Science-Fiction" not in result

        # Duplicate "Dune (Imaginary Place)" reduced to one
        assert result.count("Dune (Imaginary Place)") == 1

        # At most 5 entries
        assert len(result) <= 5

    def test_all_noise_subjects(self):
        noise = [
            "Nyt:Mass-Market-Monthly=2021-11-07",
            "Award:Hugo_Award=1966",
            "Award:Hugo_Award=Novel",
        ]
        assert clean_subjects(noise) == []

    def test_empty_subjects(self):
        assert clean_subjects([]) == []


# ---------------------------------------------------------------------------
# _build_passage identity: ONNX vs PyTorch must be byte-identical
# ---------------------------------------------------------------------------

class TestBuildPassageIdentity:

    DOCS = [
        {
            "title": "Dune",
            "authors": "Frank Herbert",
            "description": (
                '"Set on the desert planet Arrakis,\r\n\r\n'
                'Dune is the story of Paul Atreides."'
            ),
            "subjects": DUNE_SUBJECTS,
        },
        {
            "title": "Untitled",
            "authors": "",
            "description": "",
            "subjects": [],
        },
        {
            "title": "Minimal",
            "authors": "Author",
            "description": "A short desc.",
            "subjects": ["Nyt:X=2021", "Award:Y=Novel"],
        },
        {},
    ]

    def test_onnx_delegates_to_shared(self):
        """OnnxReranker._build_passage delegates to the shared build_passage."""
        from src.reranker.passage import build_passage

        onnx_inst = object.__new__(OnnxReranker)
        for doc in self.DOCS:
            assert onnx_inst._build_passage(doc) == build_passage(doc)


# ---------------------------------------------------------------------------
# Whitespace and quote normalization
# ---------------------------------------------------------------------------

class TestDescriptionNormalization:

    def test_whitespace_collapsed(self):
        doc = {
            "title": "T",
            "description": "Line one.\r\n\r\nLine two.\tTabbed.",
        }
        inst = object.__new__(OnnxReranker)
        passage = inst._build_passage(doc)
        assert "\r" not in passage
        assert "\n" not in passage
        assert "\t" not in passage
        assert "Line one. Line two. Tabbed." in passage

    def test_stray_quotes_stripped(self):
        doc = {"title": "T", "description": '"Quoted description."'}
        inst = object.__new__(OnnxReranker)
        passage = inst._build_passage(doc)
        assert passage == "T. Quoted description."

    def test_empty_after_strip(self):
        doc = {"title": "T", "description": '""'}
        inst = object.__new__(OnnxReranker)
        passage = inst._build_passage(doc)
        assert passage == "T."

    def test_subjects_omitted_when_all_noise(self):
        doc = {"title": "T", "subjects": ["Award:X=2020"]}
        inst = object.__new__(OnnxReranker)
        passage = inst._build_passage(doc)
        assert "covers" not in passage


# ---------------------------------------------------------------------------
# Print before/after for Dune (visible with -s flag)
# ---------------------------------------------------------------------------

def test_dune_passage_before_after(capsys):
    """Print before/after passage for Dune to verify the improvement."""
    doc = {
        "title": "Dune",
        "authors": "Frank Herbert",
        "description": (
            '"Set on the desert planet Arrakis,\r\n\r\n'
            "Dune is the story of the boy Paul Atreides, "
            'heir to a noble family tasked with ruling an inhospitable world."'
        ),
        "subjects": DUNE_SUBJECTS,
    }

    # Old-style passage (pipe-delimited, 300-char truncation, raw subjects)
    old_parts = []
    if doc.get("title"):
        old_parts.append(doc["title"])
    if doc.get("authors"):
        old_parts.append(f"by {doc['authors']}")
    if doc.get("description"):
        old_parts.append(doc["description"][:300])
    if doc.get("subjects") and isinstance(doc["subjects"], list):
        old_parts.append(f"Subjects: {', '.join(doc['subjects'][:5])}")
    old_passage = " | ".join(old_parts)

    # New-style passage
    inst = object.__new__(OnnxReranker)
    new_passage = inst._build_passage(doc)

    print(f"\n{'='*80}")
    print("BEFORE (old _build_passage):")
    print(f"  {old_passage}")
    print("\nAFTER (new _build_passage):")
    print(f"  {new_passage}")
    print(f"{'='*80}")

    # Structural assertions
    assert "|" not in new_passage
    assert "\r" not in new_passage
    assert "\n" not in new_passage
    assert "Nyt:" not in new_passage
    assert "Award:" not in new_passage
    assert "Fiction, Fiction" not in new_passage
