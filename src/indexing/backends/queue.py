"""Work-queue abstraction for the embedding backfill.

Two backends implement the same small contract:

* :class:`KafkaQueue`         -- portable default. A Kafka topic consumed by a
  consumer group with **manual offset commits**: an offset is committed only
  after its slice has been embedded and persisted, so a crash mid-batch causes
  the message to be redelivered (at-least-once). This is the semantics a KEDA
  ``ScaledJob`` relies on when it scales workers on consumer-group lag.
* :class:`AzureStorageQueue`  -- the original reference backend. Uses a
  visibility timeout to hide an in-flight message and deletes it on success.

Both share one interface so ``worker.py`` is backend-agnostic::

    q = get_queue()
    q.ensure()
    for msg in q.receive(max_messages=1):
        ...work...
        q.ack(msg)             # commit offset / delete message
    q.close()

Semantic notes that matter for the worker loop:

* ``receive`` blocks up to an internal receive timeout for at least one message
  and returns ``[]`` on timeout. ``idle_timeout`` tells the loop how long to
  keep polling an empty queue before deciding the backfill is drained. Kafka is
  a streaming log with no "queue empty" signal, so it uses a real idle window
  (default 30s); the Azure queue has a definite empty state, so its idle window
  is 0 and the loop exits immediately -- preserving the original behaviour.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """One queue message plus the opaque handle its backend needs to ack it."""

    content: str
    handle: Any = field(default=None, repr=False)

    def json(self) -> dict:
        return json.loads(self.content)


class MessageQueue(ABC):
    """Minimal producer/consumer contract shared by every queue backend."""

    #: Seconds the worker loop should keep polling an empty queue before it
    #: concludes the backfill is drained and exits. Backends override this.
    idle_timeout: float = 0.0

    @abstractmethod
    def ensure(self) -> None:
        """Create the topic/queue if it does not already exist (best effort)."""

    @abstractmethod
    def send(self, payload: dict) -> None:
        """Enqueue one task. ``payload`` is JSON-serialisable."""

    @abstractmethod
    def receive(self, max_messages: int = 1) -> list[Message]:
        """Return up to ``max_messages`` messages, or ``[]`` if none arrive."""

    @abstractmethod
    def ack(self, message: Message) -> None:
        """Mark a message done: commit its Kafka offset / delete it from Azure."""

    def close(self) -> None:  # pragma: no cover - trivial default
        """Flush producers and close consumers. Safe to call more than once."""


# --------------------------------------------------------------------------- #
# Kafka                                                                        #
# --------------------------------------------------------------------------- #
class KafkaQueue(MessageQueue):
    """Kafka-backed work queue using confluent-kafka (librdkafka).

    Offsets are committed one message at a time, only after ``ack``. The worker
    processes a single slice at a time (parallelism comes from running many
    workers in a consumer group), so per-message commits give clean
    at-least-once delivery without tracking a commit watermark.
    """

    def __init__(
        self,
        *,
        topic: str | None = None,
        bootstrap_servers: str | None = None,
        group_id: str | None = None,
        partitions: int | None = None,
        poll_timeout: float | None = None,
        idle_timeout: float | None = None,
    ) -> None:
        self.topic = topic or os.getenv("KAFKA_TOPIC", os.getenv("QUEUE_NAME", "embed-tasks"))
        self.bootstrap_servers = bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        self.group_id = group_id or os.getenv("KAFKA_GROUP_ID", "embed-workers")
        self.partitions = partitions if partitions is not None else int(os.getenv("KAFKA_PARTITIONS", "6"))
        self.poll_timeout = poll_timeout if poll_timeout is not None else float(os.getenv("KAFKA_POLL_TIMEOUT", "5"))
        self.idle_timeout = (
            idle_timeout if idle_timeout is not None else float(os.getenv("KAFKA_IDLE_TIMEOUT", "30"))
        )

        # Lazy client imports: importing this module must not require the SDK.
        from confluent_kafka import Consumer, Producer  # noqa: PLC0415

        self._Consumer = Consumer
        self._Producer = Producer
        self._producer = None
        self._consumer = None

    # -- producer path ----------------------------------------------------- #
    def _get_producer(self):
        if self._producer is None:
            self._producer = self._Producer(
                {
                    "bootstrap.servers": self.bootstrap_servers,
                    "enable.idempotence": True,
                    "acks": "all",
                }
            )
        return self._producer

    def send(self, payload: dict) -> None:
        producer = self._get_producer()
        # Key by batch_id so a re-enqueued slice keeps a stable partition. Order
        # across slices does not matter, but a stable key keeps retries local.
        key = str(payload.get("batch_id", "")) or None
        producer.produce(self.topic, value=json.dumps(payload).encode("utf-8"), key=key)
        producer.poll(0)

    # -- consumer path ----------------------------------------------------- #
    def _get_consumer(self):
        if self._consumer is None:
            consumer = self._Consumer(
                {
                    "bootstrap.servers": self.bootstrap_servers,
                    "group.id": self.group_id,
                    # Manual commit: an offset advances only after the slice is
                    # persisted, so a crash redelivers rather than drops work.
                    "enable.auto.commit": False,
                    # A brand-new consumer group must see the backlog that was
                    # produced before it existed, or a fresh worker would embed
                    # nothing.
                    "auto.offset.reset": "earliest",
                }
            )
            consumer.subscribe([self.topic])
            self._consumer = consumer
        return self._consumer

    def receive(self, max_messages: int = 1) -> list[Message]:
        consumer = self._get_consumer()
        out: list[Message] = []
        deadline = time.time() + self.poll_timeout
        while len(out) < max_messages:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            record = consumer.poll(timeout=remaining)
            if record is None:
                break
            if record.error():
                # Partition EOF is not an error we care about; anything else is.
                from confluent_kafka import KafkaError  # noqa: PLC0415

                if record.error().code() == KafkaError._PARTITION_EOF:
                    break
                raise RuntimeError(f"Kafka consume error: {record.error()}")
            value = record.value()
            out.append(
                Message(
                    content=value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value),
                    handle=record,
                )
            )
        return out

    def ack(self, message: Message) -> None:
        consumer = self._get_consumer()
        # Commit the record's offset+1 (the next offset to read), synchronously,
        # so the commit is durable before the worker moves on.
        consumer.commit(message=message.handle, asynchronous=False)

    def ensure(self) -> None:
        try:
            from confluent_kafka.admin import AdminClient, NewTopic  # noqa: PLC0415

            admin = AdminClient({"bootstrap.servers": self.bootstrap_servers})
            existing = admin.list_topics(timeout=10).topics
            if self.topic in existing:
                return
            futures = admin.create_topics(
                [NewTopic(self.topic, num_partitions=self.partitions, replication_factor=1)]
            )
            for fut in futures.values():
                fut.result()
            print(f"Created Kafka topic '{self.topic}' ({self.partitions} partitions)")
        except Exception as exc:  # noqa: BLE001 - topic may be auto-created / no perms
            print(f"  note: could not ensure Kafka topic '{self.topic}': {exc}")

    def close(self) -> None:
        if self._producer is not None:
            self._producer.flush(30)
        if self._consumer is not None:
            self._consumer.close()
            self._consumer = None


# --------------------------------------------------------------------------- #
# Azure Storage Queue                                                          #
# --------------------------------------------------------------------------- #
class AzureStorageQueue(MessageQueue):
    """The original Azure Storage Queue backend, kept as a reference cloud path."""

    #: The Azure queue has a definite empty state, so the loop should exit as
    #: soon as it sees one -- matching the pre-refactor behaviour.
    idle_timeout: float = 0.0

    def __init__(
        self,
        *,
        queue_name: str | None = None,
        connection_string: str | None = None,
        visibility_timeout: int | None = None,
    ) -> None:
        self.queue_name = queue_name or os.getenv("QUEUE_NAME", "embed-tasks")
        self.visibility_timeout = (
            visibility_timeout if visibility_timeout is not None else int(os.getenv("QUEUE_VISIBILITY_TIMEOUT", "3600"))
        )
        connection_string = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not connection_string:
            raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING is required for the azure queue backend")

        from azure.storage.queue import QueueServiceClient  # noqa: PLC0415

        self._client = QueueServiceClient.from_connection_string(connection_string).get_queue_client(
            self.queue_name
        )

    def ensure(self) -> None:
        try:
            self._client.create_queue()
            print(f"Created queue '{self.queue_name}'")
        except Exception:  # noqa: BLE001 - already exists
            pass

    def send(self, payload: dict) -> None:
        self._client.send_message(json.dumps(payload))

    def receive(self, max_messages: int = 1) -> list[Message]:
        raw = self._client.receive_messages(
            max_messages=max_messages, visibility_timeout=self.visibility_timeout
        )
        return [Message(content=m.content, handle=m) for m in raw]

    def ack(self, message: Message) -> None:
        self._client.delete_message(message.handle)


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #
def get_queue(backend: str | None = None, **kwargs) -> MessageQueue:
    """Construct the queue backend named by ``QUEUE_BACKEND`` (default ``kafka``)."""
    backend = (backend or os.getenv("QUEUE_BACKEND", "kafka")).strip().lower()
    if backend == "kafka":
        return KafkaQueue(**kwargs)
    if backend in ("azure", "azure-queue", "storage-queue"):
        return AzureStorageQueue(**kwargs)
    raise ValueError(f"Unknown QUEUE_BACKEND '{backend}' (expected 'kafka' or 'azure')")
