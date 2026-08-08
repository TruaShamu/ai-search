"""Assemble dense embedding shards into the vector array + metadata that load.py consumes.

The embed workers each handle one slice of the corpus and write a compressed
``.npz`` shard containing the dense vectors plus the ``work_id`` of every row.
This script stitches those shards back together.

Why it keys on ``work_id`` rather than position: a worker skips books that are
not tier<=1 or that have no description, so a shard covering corpus indices
[1000, 1500) may contain fewer than 500 vectors. Rebuilding by slice offset
would therefore drift out of alignment with the corpus and silently attach the
wrong metadata to a vector. Every row is matched back to its source record by
``work_id`` and the script refuses to write anything if that mapping is not
exact.

Usage (cloud shards):
    python -m src.indexing.assemble --corpus data/processed/books_goodreads_v2.jsonl

Usage (local fallback shards):
    python -m src.indexing.assemble --local-shards data/shards \
        --corpus data/processed/books_goodreads_v2.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "data" / "processed" / "books_goodreads_v2.jsonl"
DEFAULT_OUT_DIR = ROOT / "data" / "index"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="JSONL corpus the shards were built from")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Where to write embeddings.npy + metadata.jsonl")
    parser.add_argument("--shard-prefix", default="shards", help="Object-store prefix for shards")
    parser.add_argument("--local-shards", type=Path, default=None, help="Read .npz shards from this directory instead of the object store")
    parser.add_argument("--dim", type=int, default=256, help="Expected embedding dimension")
    parser.add_argument("--allow-partial", action="store_true", help="Proceed even if some eligible corpus records have no vector")
    return parser.parse_args()


def load_local_shards(directory: Path) -> list[tuple[int, np.ndarray, list[str]]]:
    shards = []
    paths = sorted(directory.glob("*.npz"))
    if not paths:
        sys.exit(f"No .npz shards found in {directory}")
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            shards.append((int(data["start_idx"][0]), data["embeddings"], [str(w) for w in data["work_ids"]]))
        print(f"  read {path.name}: {shards[-1][1].shape[0]} vectors @ start_idx={shards[-1][0]}")
    return shards


def load_object_store_shards(prefix: str) -> list[tuple[int, np.ndarray, list[str]]]:
    """Download every ``.npz`` shard under ``prefix`` from the configured store.

    Backend is chosen by ``OBJECT_STORE_BACKEND`` (s3/MinIO by default, azure as
    the reference cloud path) -- the same abstraction the worker writes through.
    """
    from src.indexing.backends import get_object_store

    store = get_object_store()
    names = [n for n in store.list(f"{prefix}/") if n.endswith(".npz")]
    if not names:
        sys.exit(f"No shards found under {prefix}/")
    print(f"Found {len(names)} shards under {prefix}/")

    shards = []
    for i, name in enumerate(names, 1):
        raw = store.get_bytes(name)
        with np.load(io.BytesIO(raw), allow_pickle=True) as data:
            shards.append((int(data["start_idx"][0]), data["embeddings"], [str(w) for w in data["work_ids"]]))
        if i % 25 == 0 or i == len(names):
            print(f"  downloaded {i}/{len(names)} shards")
    return shards


def load_corpus(path: Path) -> tuple[dict[str, dict], int]:
    """Return work_id -> record, plus the count of records a worker would embed."""
    by_work_id: dict[str, dict] = {}
    eligible = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            work_id = record.get("work_id", "")
            if work_id:
                by_work_id[work_id] = record
            if record.get("tier", 99) <= 1 and record.get("description"):
                eligible += 1
    return by_work_id, eligible


def main() -> None:
    args = parse_args()

    print(f"Loading corpus from {args.corpus}...")
    corpus, eligible = load_corpus(args.corpus)
    print(f"  {len(corpus)} records, {eligible} eligible for embedding")

    if args.local_shards:
        print(f"Reading shards from {args.local_shards}...")
        shards = load_local_shards(args.local_shards)
    else:
        shards = load_object_store_shards(args.shard_prefix)

    # Deterministic order: by the slice offset the worker was given.
    shards.sort(key=lambda s: s[0])

    vectors: list[np.ndarray] = []
    work_ids: list[str] = []
    for start_idx, embeddings, shard_work_ids in shards:
        if embeddings.shape[0] != len(shard_work_ids):
            sys.exit(f"Shard @{start_idx} is malformed: {embeddings.shape[0]} vectors vs {len(shard_work_ids)} work_ids")
        vectors.append(embeddings)
        work_ids.extend(shard_work_ids)

    matrix = np.vstack(vectors).astype(np.float32)
    print(f"\nStitched {matrix.shape[0]} vectors of dim {matrix.shape[1]} from {len(shards)} shards")

    if matrix.shape[1] != args.dim:
        sys.exit(f"Dimension mismatch: shards are {matrix.shape[1]}d but --dim is {args.dim}")

    # A duplicate work_id means a slice was enqueued or processed twice. Left
    # unchecked it inflates the index and corrupts the id->metadata mapping.
    duplicates = len(work_ids) - len(set(work_ids))
    if duplicates:
        from collections import Counter

        dupes = sorted(w for w, n in Counter(work_ids).items() if n > 1)
        sys.exit(f"{duplicates} duplicate work_ids across shards (e.g. {dupes[:5]}). Re-run the backfill cleanly.")

    missing = [w for w in work_ids if w not in corpus]
    if missing:
        sys.exit(f"{len(missing)} shard work_ids are absent from the corpus (e.g. {missing[:5]}). Wrong corpus file?")

    coverage = matrix.shape[0] / eligible if eligible else 0.0
    print(f"Coverage: {matrix.shape[0]}/{eligible} eligible records ({coverage:.1%})")
    if matrix.shape[0] != eligible and not args.allow_partial:
        sys.exit(
            f"Missing {eligible - matrix.shape[0]} vectors. Some slices never completed. "
            "Re-enqueue them, or pass --allow-partial to index what exists."
        )

    norms = np.linalg.norm(matrix, axis=1)
    print(f"Vector norms: min={norms.min():.4f} max={norms.max():.4f} (expect ~1.0 for cosine)")
    if not np.allclose(norms, 1.0, atol=1e-2):
        sys.exit("Vectors are not unit-normalised; inner-product search would be wrong.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = args.out_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for work_id in work_ids:
            f.write(json.dumps(corpus[work_id], ensure_ascii=False) + "\n")
    print(f"Wrote {metadata_path} ({len(work_ids)} rows)")

    vectors_path = args.out_dir / "embeddings.npy"
    np.save(vectors_path, matrix)
    print(f"Wrote {vectors_path} ({matrix.shape[0]} vectors, dim {matrix.shape[1]})")

    # Round-trip so a corrupt write is caught here rather than in load.py.
    check = np.load(vectors_path, mmap_mode="r")
    assert check.shape == matrix.shape, f"Round-trip shape mismatch: {check.shape} vs {matrix.shape}"
    assert check.shape[0] == len(work_ids), f"Round-trip mismatch: {check.shape[0]} vs {len(work_ids)}"
    print("\nRound-trip verified. Next: python -m src.indexing.load --recreate --qdrant-url <url>")


if __name__ == "__main__":
    main()
