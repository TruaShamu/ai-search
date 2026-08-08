"""OpenTelemetry tracing — optional, non-invasive, off by default.

Tracing activates only when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set. With it
unset (production without a collector, the test suite, a clean clone) every
helper here is a cheap no-op and none of the ``opentelemetry-*`` packages need
to be installed. When an endpoint *is* configured, spans are exported over
OTLP/HTTP to a collector — Jaeger in ``docker-compose.yml``.

Call sites stay clean:

    from src.telemetry import span

    with span("retrieval.qdrant_query", **{"search.mode": mode}):
        ...

The context manager yields the live span (or ``None`` when tracing is off), so
attribute-setting on the yielded object is optional and always safe.

Distributed traces cross the Kafka work queue via :func:`inject_context` /
:func:`extract_context`, which serialise the current trace context into message
headers and restore it in the consumer, so an embedding backfill is one trace
from the enqueue call through every worker that handles a slice.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_configured = False


def telemetry_enabled() -> bool:
    """True when an OTLP endpoint is configured and the SDK is not disabled."""
    if os.getenv("OTEL_SDK_DISABLED", "").strip().lower() == "true":
        return False
    return bool(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip())


def setup_telemetry(service_name: str, *, fastapi_app=None) -> bool:
    """Configure the global tracer provider once. Idempotent and best-effort.

    Returns whether tracing ended up active. Safe to call when the OTLP endpoint
    is unset (returns ``False`` without touching anything) or when the
    ``opentelemetry`` packages are missing (logs a warning, returns ``False``).
    """
    global _configured
    if _configured:
        return True
    if not telemetry_enabled():
        return False

    try:
        from opentelemetry import trace  # noqa: PLC0415
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: PLC0415
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
            OTLPSpanExporter,
        )
    except ImportError as exc:
        logger.warning(
            "OTEL_EXPORTER_OTLP_ENDPOINT is set but opentelemetry packages are "
            "missing (%s); tracing stays off.",
            exc,
        )
        return False

    resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", service_name)})
    provider = TracerProvider(resource=resource)
    # Exporter is selectable via the standard OTEL_TRACES_EXPORTER var:
    #   otlp (default) -> OTLP/HTTP to OTEL_EXPORTER_OTLP_ENDPOINT (Jaeger)
    #   console        -> print spans (local debugging)
    #   none           -> configure the provider but export nothing (tests)
    # BatchSpanProcessor exports off the request path, so span export never adds
    # latency to a search or an embed.
    exporter_name = os.getenv("OTEL_TRACES_EXPORTER", "otlp").strip().lower()
    if exporter_name == "none":
        pass
    elif exporter_name == "console":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter  # noqa: PLC0415

        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)

    # Auto-instrument outbound HTTP (Azure OpenAI, Qdrant REST) so those calls
    # appear as child spans without touching their call sites.
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor  # noqa: PLC0415

        HTTPXClientInstrumentor().instrument()
    except Exception as exc:  # noqa: BLE001 - optional add-on
        logger.debug("httpx instrumentation unavailable: %s", exc)

    if fastapi_app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # noqa: PLC0415

            FastAPIInstrumentor.instrument_app(fastapi_app)
        except Exception as exc:  # noqa: BLE001 - optional add-on
            logger.debug("fastapi instrumentation unavailable: %s", exc)

    _configured = True
    logger.info(
        "OpenTelemetry tracing enabled for %s -> %s",
        service_name,
        os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
    )
    return True


@contextmanager
def span(name: str, **attributes):
    """Start a span if tracing is active; otherwise a zero-cost no-op.

    Attribute values that are ``None`` are skipped so callers can pass optional
    fields directly. Yields the span (or ``None`` when tracing is off).
    """
    if not _configured:
        yield None
        return

    from opentelemetry import trace  # noqa: PLC0415

    tracer = trace.get_tracer("booksearch")
    with tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        yield current


# --------------------------------------------------------------------------- #
# Cross-process propagation over the Kafka work queue                          #
# --------------------------------------------------------------------------- #
def inject_context() -> dict[str, str]:
    """Serialise the current trace context into a header dict (W3C traceparent).

    Returns ``{}`` when tracing is off, so producers can always call it and pass
    the result straight to Kafka message headers.
    """
    if not _configured:
        return {}
    from opentelemetry.propagate import inject  # noqa: PLC0415

    carrier: dict[str, str] = {}
    inject(carrier)
    return carrier


def extract_context(headers: dict[str, str] | None):
    """Restore a trace context from message headers, or ``None`` if unavailable.

    The returned object is an OTel ``Context`` suitable to pass as the ``context``
    of a new span so the consumer's work continues the producer's trace.
    """
    if not _configured or not headers:
        return None
    from opentelemetry.propagate import extract  # noqa: PLC0415

    return extract(headers)


@contextmanager
def span_with_context(name: str, context, **attributes):
    """Like :func:`span`, but rooted at an extracted remote ``context``.

    Used by the worker to attach a slice's processing span to the producer trace
    carried in the Kafka headers. Falls back to a no-op when tracing is off.
    """
    if not _configured:
        yield None
        return

    from opentelemetry import trace  # noqa: PLC0415

    tracer = trace.get_tracer("booksearch")
    with tracer.start_as_current_span(name, context=context) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        yield current
