"""Pluggable infrastructure backends for the embedding pipeline.

The embedding backfill talks to two pieces of infrastructure: a work *queue*
that hands out slice tasks, and an *object store* that holds the input slices,
the dense shards, and failure diagnostics. Both were originally hard-wired to
Azure (Storage Queue + Blob). They are now addressed through small interfaces so
the same pipeline runs on Kafka + S3/MinIO (the portable default) or on Azure
(the reference cloud deployment), selected entirely by environment variables.

    QUEUE_BACKEND         kafka (default) | azure
    OBJECT_STORE_BACKEND  s3 (default)    | azure

Nothing here imports a backend's client library at module import time -- each
concrete backend imports its SDK lazily inside ``__init__`` -- so importing this
package (and running the test suite) never requires confluent-kafka, boto3, or
the azure SDKs to be installed.
"""

from src.indexing.backends.objectstore import ObjectStore, get_object_store
from src.indexing.backends.queue import Message, MessageQueue, get_queue

__all__ = [
    "Message",
    "MessageQueue",
    "ObjectStore",
    "get_object_store",
    "get_queue",
]
