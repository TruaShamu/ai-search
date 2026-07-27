"""Tests for reranker passage building and subject cleaning."""

import pytest

from src.reranker.onnx_reranker import OnnxReranker, _clean_subjects as onnx_clean
from src.reranker.model import CrossEncoderReranker, _clean_subjects as pt_clean


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
        result = onnx_clean(DUNE_SUBJECTS)

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
        assert onnx_clean(noise) == []

    def test_empty_subjects(self):
        assert onnx_clean([]) == []

    def test_onnx_and_pytorch_clean_subjects_match(self):
        """Both backends must produce identical subject cleaning."""
        cases = [
            DUNE_SUBJECTS,
            [],
            ["Fiction"],
            ["A, B", "B, C"],
            ["Nyt:X=2021"],
        ]
        for subjects in cases:
            assert onnx_clean(subjects) == pt_clean(subjects), (
                f"Diverged on {subjects}"
            )


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

    def test_passages_identical(self):
        """Both backends must produce byte-identical passages."""
        onnx_inst = object.__new__(OnnxReranker)
        pt_inst = object.__new__(CrossEncoderReranker)

        for doc in self.DOCS:
            onnx_p = onnx_inst._build_passage(doc)
            pt_p = pt_inst._build_passage(doc)
            assert onnx_p == pt_p, (
                f"Passages diverge for {doc.get('title', '(empty)')!r}:\n"
                f"  ONNX:    {onnx_p!r}\n"
                f"  PyTorch: {pt_p!r}"
            )


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
    print(f"\nAFTER (new _build_passage):")
    print(f"  {new_passage}")
    print(f"{'='*80}")

    # Structural assertions
    assert "|" not in new_passage
    assert "\r" not in new_passage
    assert "\n" not in new_passage
    assert "Nyt:" not in new_passage
    assert "Award:" not in new_passage
    assert "Fiction, Fiction" not in new_passage
