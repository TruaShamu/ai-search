"""Unit tests for the pluggable queue / object-store backends and the
backend-agnostic worker loop.

These tests never import confluent-kafka, boto3, or the azure SDKs: they cover
the factory dispatch, the message contract, and the worker's at-least-once
orchestration through in-memory fakes. The concrete Kafka/S3/Azure backends
import their SDKs lazily, so a clean clone with none of them installed still
runs this file.
"""

import json

import numpy as np
import pytest

import src.indexing.worker as worker
from src.indexing.backends import get_object_store, get_queue
from src.indexing.backends.queue import Message


# --------------------------------------------------------------------------- #
# Factory dispatch                                                             #
# --------------------------------------------------------------------------- #
def test_get_queue_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown QUEUE_BACKEND"):
        get_queue("rabbitmq")


def test_get_object_store_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown OBJECT_STORE_BACKEND"):
        get_object_store("gcs")


def test_message_json_roundtrips():
    payload = {"batch_id": "batch-0007", "start_idx": 0, "end_idx": 10}
    msg = Message(content=json.dumps(payload))
    assert msg.json() == payload


# --------------------------------------------------------------------------- #
# In-memory fakes implementing the backend contracts                          #
# --------------------------------------------------------------------------- #
class FakeQueue:
    """A minimal MessageQueue: a list of pending messages plus ack tracking."""

    idle_timeout = 0.0

    def __init__(self, payloads=None):
        self.pending = [Message(content=json.dumps(p), handle=p) for p in (payloads or [])]
        self.sent = []
        self.acked = []
        self.ensured = False
        self.closed = False

    def ensure(self):
        self.ensured = True

    def send(self, payload):
        self.sent.append(payload)

    def receive(self, max_messages=1):
        out = []
        while self.pending and len(out) < max_messages:
            out.append(self.pending.pop(0))
        return out

    def ack(self, message):
        self.acked.append(message.json()["batch_id"])

    def close(self):
        self.closed = True


class FakeStore:
    """A dict-backed ObjectStore. Keys in ``fail_reads`` raise on get_bytes."""

    def __init__(self, seed=None, fail_reads=()):
        self.objects = dict(seed or {})
        self.fail_reads = set(fail_reads)

    def put_bytes(self, key, data):
        self.objects[key] = data

    def get_bytes(self, key):
        if key in self.fail_reads:
            raise RuntimeError(f"simulated read failure for {key}")
        return self.objects[key]

    def list(self, prefix):
        return sorted(k for k in self.objects if k.startswith(prefix))

    def exists(self, key):
        return key in self.objects


def _slice_bytes(*books):
    return ("\n".join(json.dumps(b) for b in books) + "\n").encode("utf-8")


def _tier1_book(work_id):
    return {"work_id": work_id, "title": f"Title {work_id}", "authors": ["A"],
            "description": "a readable description", "tier": 1}


@pytest.fixture
def patched_embed(monkeypatch):
    """Stub out the model + encoder so no torch / network is needed."""
    monkeypatch.setattr(worker, "get_model", lambda: object())
    monkeypatch.setattr(worker, "embed_books", lambda books, dim=256: np.ones((len(books), 4), dtype=np.float32))


# --------------------------------------------------------------------------- #
# Worker loop: at-least-once semantics                                         #
# --------------------------------------------------------------------------- #
def test_worker_processes_and_acks_every_slice(monkeypatch, patched_embed):
    queue = FakeQueue([
        {"batch_id": "batch-0000", "blob_path": "inputs/slices/batch-0000.jsonl", "start_idx": 0, "end_idx": 2},
        {"batch_id": "batch-0001", "blob_path": "inputs/slices/batch-0001.jsonl", "start_idx": 2, "end_idx": 4},
    ])
    store = FakeStore({
        "inputs/slices/batch-0000.jsonl": _slice_bytes(_tier1_book("w0"), _tier1_book("w1")),
        "inputs/slices/batch-0001.jsonl": _slice_bytes(_tier1_book("w2"), _tier1_book("w3")),
    })
    monkeypatch.setattr(worker, "get_queue", lambda: queue)
    monkeypatch.setattr(worker, "get_object_store", lambda: store)

    failures = worker.worker_loop(loop=True)

    assert failures == 0
    assert queue.acked == ["batch-0000", "batch-0001"]
    # A dense shard was written for each slice.
    assert store.exists("shards/batch-0000.npz")
    assert store.exists("shards/batch-0001.npz")
    assert queue.closed is True


def test_worker_does_not_ack_on_failure(monkeypatch, patched_embed):
    """A slice that fails mid-flight must stay un-acked so it is redelivered."""
    queue = FakeQueue([
        {"batch_id": "batch-0009", "blob_path": "inputs/slices/batch-0009.jsonl", "start_idx": 0, "end_idx": 2},
    ])
    store = FakeStore(fail_reads={"inputs/slices/batch-0009.jsonl"})
    monkeypatch.setattr(worker, "get_queue", lambda: queue)
    monkeypatch.setattr(worker, "get_object_store", lambda: store)

    failures = worker.worker_loop(loop=True)

    assert failures == 1
    assert queue.acked == []  # never acked -> at-least-once redelivery
    # The failure was persisted for offline diagnosis.
    assert any(k.startswith("errors/batch-0009-") for k in store.objects)


def test_worker_startup_aborts_without_consuming_if_model_fails(monkeypatch):
    queue = FakeQueue([
        {"batch_id": "batch-0000", "blob_path": "x", "start_idx": 0, "end_idx": 1},
    ])
    store = FakeStore()

    def boom():
        raise RuntimeError("model download failed")

    monkeypatch.setattr(worker, "get_queue", lambda: queue)
    monkeypatch.setattr(worker, "get_object_store", lambda: store)
    monkeypatch.setattr(worker, "get_model", boom)

    failures = worker.worker_loop(loop=True)

    assert failures == 1
    assert queue.acked == []
    assert queue.pending  # the message was never received
    assert any(k.startswith("errors/startup-") for k in store.objects)


# --------------------------------------------------------------------------- #
# Producer path                                                               #
# --------------------------------------------------------------------------- #
def test_enqueue_uploads_slices_and_sends_tasks(monkeypatch, tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps(_tier1_book(f"w{i}")) for i in range(5)) + "\n", encoding="utf-8")

    queue = FakeQueue()
    store = FakeStore()
    monkeypatch.setattr(worker, "get_queue", lambda: queue)
    monkeypatch.setattr(worker, "get_object_store", lambda: store)
    monkeypatch.setattr(worker, "BATCH_CHUNK_SIZE", 2)

    worker.enqueue_batches(str(corpus))

    # 5 books, chunk size 2 -> 3 slices, each uploaded and enqueued.
    assert len(queue.sent) == 3
    assert store.exists("inputs/slices/batch-0000.jsonl")
    assert store.exists("inputs/slices/batch-0002.jsonl")
    batch_ids = [m["batch_id"] for m in queue.sent]
    assert batch_ids == ["batch-0000", "batch-0001", "batch-0002"]
    # Absolute indexing is preserved across the corpus.
    assert queue.sent[-1]["start_idx"] == 4 and queue.sent[-1]["end_idx"] == 5
