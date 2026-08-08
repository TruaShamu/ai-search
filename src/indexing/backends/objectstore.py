"""Object-store abstraction for the embedding pipeline's slices and shards.

The backfill reads pre-cut corpus slices and writes dense ``.npz`` shards plus
failure diagnostics. That IO used to be Azure Blob only; it now goes through a
tiny interface with two backends:

* :class:`AzureBlobStore`  -- default. The managed Azure Blob container the
  deployment runs against; locally, the Azurite emulator stands in for it.
* :class:`S3ObjectStore`   -- optional, portable. Talks to any S3-compatible
  endpoint via boto3, so the same code can address AWS S3 or MinIO by setting
  ``S3_ENDPOINT_URL``. Kept to demonstrate the abstraction is not Azure-bound.

Both expose four operations the pipeline needs::

    store = get_object_store()
    store.put_bytes("inputs/slices/batch-0001.jsonl", data)
    raw = store.get_bytes("inputs/slices/batch-0001.jsonl")
    names = store.list("shards/")
    exists = store.exists("shards/batch-0001.npz")

Keys are always forward-slash paths; the S3 backend maps them straight to object
keys, the Azure backend to blob names.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class ObjectStore(ABC):
    """Minimal blob/object contract shared by every object-store backend."""

    @abstractmethod
    def put_bytes(self, key: str, data: bytes) -> None:
        """Write ``data`` at ``key``, overwriting any existing object."""

    @abstractmethod
    def get_bytes(self, key: str) -> bytes:
        """Read and return the object at ``key``."""

    @abstractmethod
    def list(self, prefix: str) -> list[str]:
        """Return every key under ``prefix`` (sorted)."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return whether an object exists at ``key``."""


# --------------------------------------------------------------------------- #
# S3 / MinIO                                                                   #
# --------------------------------------------------------------------------- #
class S3ObjectStore(ObjectStore):
    """S3-compatible object store (AWS S3 or MinIO) via boto3.

    ``S3_ENDPOINT_URL`` selects the target: unset for AWS S3, or the MinIO
    service URL (e.g. ``http://minio:9000``) in cluster / compose. The bucket is
    created on first use so a fresh MinIO needs no manual setup.
    """

    def __init__(
        self,
        *,
        bucket: str | None = None,
        endpoint_url: str | None = None,
        region: str | None = None,
    ) -> None:
        self.bucket = bucket or os.getenv("S3_BUCKET", "embeddings")
        endpoint_url = endpoint_url or os.getenv("S3_ENDPOINT_URL") or None
        region = region or os.getenv("AWS_REGION", "us-east-1")

        import boto3  # noqa: PLC0415
        from botocore.config import Config  # noqa: PLC0415

        # Path-style addressing is required by MinIO and harmless on AWS.
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region,
            config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"),
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        from botocore.exceptions import ClientError  # noqa: PLC0415

        try:
            self._client.head_bucket(Bucket=self.bucket)
        except ClientError:
            try:
                self._client.create_bucket(Bucket=self.bucket)
            except ClientError as exc:  # already exists / race
                if exc.response.get("Error", {}).get("Code") not in (
                    "BucketAlreadyOwnedByYou",
                    "BucketAlreadyExists",
                ):
                    raise

    def put_bytes(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def get_bytes(self, key: str) -> bytes:
        return self._client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def list(self, prefix: str) -> list[str]:
        keys: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))
        return sorted(keys)

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError  # noqa: PLC0415

        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False


# --------------------------------------------------------------------------- #
# Azure Blob                                                                   #
# --------------------------------------------------------------------------- #
class AzureBlobStore(ObjectStore):
    """The original Azure Blob container, kept as a reference cloud path."""

    def __init__(
        self,
        *,
        container: str | None = None,
        connection_string: str | None = None,
    ) -> None:
        self.container = container or os.getenv("STORAGE_CONTAINER", "embeddings")
        connection_string = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not connection_string:
            raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING is required for the azure object store")

        from azure.storage.blob import BlobServiceClient  # noqa: PLC0415

        self._client = BlobServiceClient.from_connection_string(connection_string).get_container_client(
            self.container
        )
        try:
            self._client.create_container()
        except Exception:  # noqa: BLE001 - already exists
            pass

    def put_bytes(self, key: str, data: bytes) -> None:
        self._client.upload_blob(name=key, data=data, overwrite=True)

    def get_bytes(self, key: str) -> bytes:
        return self._client.download_blob(key).readall()

    def list(self, prefix: str) -> list[str]:
        return sorted(b.name for b in self._client.list_blobs(name_starts_with=prefix))

    def exists(self, key: str) -> bool:
        return self._client.get_blob_client(key).exists()


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #
def get_object_store(backend: str | None = None, **kwargs) -> ObjectStore:
    """Construct the store named by ``OBJECT_STORE_BACKEND`` (default ``azure``)."""
    backend = (backend or os.getenv("OBJECT_STORE_BACKEND", "azure")).strip().lower()
    if backend in ("azure", "azure-blob", "blob"):
        return AzureBlobStore(**kwargs)
    if backend in ("s3", "minio"):
        return S3ObjectStore(**kwargs)
    raise ValueError(f"Unknown OBJECT_STORE_BACKEND '{backend}' (expected 'azure' or 's3')")
