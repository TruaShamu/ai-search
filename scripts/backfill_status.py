"""Status probe for the sharded embedding backfill.

Reports queue depth, shard count, and estimated completion so the backfill can
be watched without shelling out to `az` (whose CLI parsing mangles storage
connection strings on Windows).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from azure.storage.blob import BlobServiceClient
from azure.storage.queue import QueueServiceClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="embeddings")
    parser.add_argument("--prefix", default="shards")
    parser.add_argument("--queue", default="embed-tasks")
    parser.add_argument("--expected", type=int, default=0, help="Expected total shard count")
    args = parser.parse_args()

    conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        sys.exit("Set AZURE_STORAGE_CONNECTION_STRING")

    queue = QueueServiceClient.from_connection_string(conn).get_queue_client(args.queue)
    props = queue.get_queue_properties()
    depth = props.approximate_message_count

    container = BlobServiceClient.from_connection_string(conn).get_container_client(args.container)
    shards = list(container.list_blobs(name_starts_with=f"{args.prefix}/"))
    total_bytes = sum(b.size for b in shards)

    print(f"queue '{args.queue}': ~{depth} messages visible")
    print(f"shards: {len(shards)} ({total_bytes / 1e6:.1f} MB)")
    if args.expected:
        print(f"progress: {len(shards)}/{args.expected} ({len(shards) / args.expected:.1%})")
    if shards:
        newest = max(shards, key=lambda b: b.last_modified)
        print(f"newest shard: {newest.name} @ {newest.last_modified:%H:%M:%S}")


if __name__ == "__main__":
    main()
