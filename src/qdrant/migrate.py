from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from qdrant_client import QdrantClient, models
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "index"
FAISS_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.jsonl"
VECTORIZER_PATH = INDEX_DIR / "tfidf_vectorizer.pkl"
BATCH_SIZE = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate local FAISS index into Qdrant.")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant base URL")
    parser.add_argument("--collection", default="books", help="Qdrant collection name")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate the collection")
    return parser.parse_args()


def load_metadata(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def combined_text(book: dict) -> str:
    subjects = book.get("subjects") or []
    if not isinstance(subjects, list):
        subjects = [str(subjects)]
    parts = [
        book.get("title") or "",
        book.get("description") or "",
        " ".join(subjects),
    ]
    return " ".join(part for part in parts if part).strip()


def build_payload(book: dict, point_id: int) -> dict:
    authors = book.get("authors") or []
    if not isinstance(authors, list):
        authors = [str(authors)]

    payload = dict(book)
    payload["id"] = point_id
    payload["authors"] = ", ".join(author for author in authors if author)
    payload["year"] = book.get("first_publish_year")
    payload["cover_url"] = (
        f"https://covers.openlibrary.org/b/id/{book['cover_id']}-M.jpg"
        if book.get("cover_id")
        else None
    )
    return payload


def load_dense_vectors(path: Path) -> np.ndarray:
    index = faiss.read_index(str(path))
    if hasattr(index, "reconstruct_n"):
        vectors = index.reconstruct_n(0, index.ntotal)
    else:
        vectors = np.vstack([index.reconstruct(i) for i in range(index.ntotal)])
    return np.asarray(vectors, dtype=np.float32)


def save_vectorizer(vectorizer: TfidfVectorizer, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(vectorizer, f)


def ensure_collection(client: QdrantClient, collection: str, dense_dim: int, recreate: bool) -> None:
    exists = client.collection_exists(collection_name=collection)

    if exists and recreate:
        client.delete_collection(collection_name=collection)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=collection,
            vectors_config={
                "dense": models.VectorParams(size=dense_dim, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(),
            },
        )
        client.create_payload_index(
            collection_name=collection,
            field_name="year",
            field_schema=models.PayloadSchemaType.INTEGER,
        )
        client.create_payload_index(
            collection_name=collection,
            field_name="tier",
            field_schema=models.PayloadSchemaType.INTEGER,
        )


def upload_points(
    client: QdrantClient,
    collection: str,
    metadata: list[dict],
    dense_vectors: np.ndarray,
    sparse_matrix,
) -> None:
    for batch_start in tqdm(range(0, len(metadata), BATCH_SIZE), desc="Uploading", unit="batch"):
        batch_end = min(batch_start + BATCH_SIZE, len(metadata))
        points: list[models.PointStruct] = []

        for point_id in range(batch_start, batch_end):
            sparse_row = sparse_matrix.getrow(point_id)
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vectors[point_id].tolist(),
                        "sparse": models.SparseVector(
                            indices=sparse_row.indices.tolist(),
                            values=sparse_row.data.astype(float).tolist(),
                        ),
                    },
                    payload=build_payload(metadata[point_id], point_id),
                )
            )

        client.upsert(collection_name=collection, points=points, wait=True)


def main() -> None:
    args = parse_args()

    print(f"Loading metadata from {METADATA_PATH}...")
    metadata = load_metadata(METADATA_PATH)

    print(f"Loading dense vectors from {FAISS_PATH}...")
    dense_vectors = load_dense_vectors(FAISS_PATH)

    if len(metadata) != dense_vectors.shape[0]:
        raise ValueError(
            f"Metadata/vector count mismatch: {len(metadata)} metadata rows vs "
            f"{dense_vectors.shape[0]} vectors."
        )

    print("Fitting TF-IDF vectorizer...")
    documents = [combined_text(book) for book in metadata]
    vectorizer = TfidfVectorizer(dtype=np.float32)
    sparse_matrix = vectorizer.fit_transform(documents).tocsr()
    save_vectorizer(vectorizer, VECTORIZER_PATH)
    print(f"Saved TF-IDF vectorizer to {VECTORIZER_PATH}")

    print(f"Connecting to Qdrant at {args.qdrant_url}...")
    client = QdrantClient(url=args.qdrant_url)

    print(f"Ensuring collection '{args.collection}' exists...")
    ensure_collection(
        client=client,
        collection=args.collection,
        dense_dim=dense_vectors.shape[1],
        recreate=args.recreate,
    )

    print(f"Uploading {len(metadata):,} points to Qdrant...")
    upload_points(
        client=client,
        collection=args.collection,
        metadata=metadata,
        dense_vectors=dense_vectors,
        sparse_matrix=sparse_matrix,
    )
    print("Migration complete.")


if __name__ == "__main__":
    main()
