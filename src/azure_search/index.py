"""Create and manage Azure AI Search index for book search."""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
API_KEY = os.environ["AZURE_SEARCH_ADMIN_KEY"]
INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX", "books-v1")
API_VERSION = "2024-07-01"

HEADERS = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}


def create_index():
    """Create the books index with hybrid search (BM25 + vector) configuration."""
    url = f"{ENDPOINT}/indexes/{INDEX_NAME}?api-version={API_VERSION}"

    schema = {
        "name": INDEX_NAME,
        "fields": [
            {"name": "id", "type": "Edm.String", "key": True, "filterable": True},
            {
                "name": "title",
                "type": "Edm.String",
                "searchable": True,
                "retrievable": True,
                "analyzer": "en.lucene",
            },
            {
                "name": "authors",
                "type": "Edm.String",
                "searchable": True,
                "retrievable": True,
            },
            {
                "name": "description",
                "type": "Edm.String",
                "searchable": True,
                "retrievable": True,
                "analyzer": "en.lucene",
            },
            {
                "name": "subjects",
                "type": "Collection(Edm.String)",
                "searchable": True,
                "filterable": True,
                "facetable": True,
            },
            {
                "name": "year",
                "type": "Edm.Int32",
                "filterable": True,
                "sortable": True,
                "retrievable": True,
            },
            {
                "name": "cover_url",
                "type": "Edm.String",
                "retrievable": True,
                "searchable": False,
                "filterable": False,
            },
            {
                "name": "work_id",
                "type": "Edm.String",
                "retrievable": True,
                "filterable": True,
            },
            {
                "name": "tier",
                "type": "Edm.Int32",
                "filterable": True,
                "retrievable": True,
            },
            {
                "name": "embedding",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "retrievable": False,
                "dimensions": 256,
                "vectorSearchProfile": "hnsw-nomic",
            },
        ],
        "vectorSearch": {
            "algorithms": [
                {
                    "name": "hnsw-algo",
                    "kind": "hnsw",
                    "hnswParameters": {
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine",
                    },
                }
            ],
            "profiles": [
                {
                    "name": "hnsw-nomic",
                    "algorithm": "hnsw-algo",
                }
            ],
        },
    }

    resp = requests.put(url, headers=HEADERS, json=schema)
    if resp.status_code in (200, 201, 204):
        print(f"[OK] Index '{INDEX_NAME}' created/updated successfully")
        return True
    else:
        print(f"[FAIL] Create index: {resp.status_code}")
        print(resp.text[:500])
        return False


def delete_index():
    """Delete the index (for clean re-creation)."""
    url = f"{ENDPOINT}/indexes/{INDEX_NAME}?api-version={API_VERSION}"
    resp = requests.delete(url, headers=HEADERS)
    if resp.status_code == 204:
        print(f"✓ Index '{INDEX_NAME}' deleted")
    elif resp.status_code == 404:
        print(f"  Index '{INDEX_NAME}' doesn't exist")
    else:
        print(f"✗ Delete failed: {resp.status_code} - {resp.text}")


def get_index_stats():
    """Get document count and storage size."""
    url = f"{ENDPOINT}/indexes/{INDEX_NAME}/stats?api-version={API_VERSION}"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code == 200:
        stats = resp.json()
        print(f"Documents: {stats['documentCount']}")
        print(f"Storage:   {stats['storageSize'] / 1024 / 1024:.1f} MB")
        return stats
    else:
        print(f"✗ Stats failed: {resp.status_code}")
        return None


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "create"
    if cmd == "create":
        create_index()
    elif cmd == "delete":
        delete_index()
    elif cmd == "stats":
        get_index_stats()
    elif cmd == "recreate":
        delete_index()
        create_index()
    else:
        print("Usage: python -m src.azure_search.index [create|delete|stats|recreate]")
