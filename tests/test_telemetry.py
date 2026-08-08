"""Tests for the optional OpenTelemetry tracing layer (src/telemetry.py).

The layer must be a zero-cost no-op unless OTEL_EXPORTER_OTLP_ENDPOINT is set,
so the API/worker/eval code can call span()/inject_context() unconditionally
without pulling in the opentelemetry SDK. These tests exercise the disabled path
(always) and, when the SDK is installed, a live setup + Kafka header round-trip.
"""

import importlib

import pytest


@pytest.fixture
def fresh_telemetry(monkeypatch):
    """Reload src.telemetry with a clean module-level _configured flag."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    import src.telemetry as telemetry

    importlib.reload(telemetry)
    return telemetry


def test_disabled_when_endpoint_unset(fresh_telemetry):
    assert fresh_telemetry.telemetry_enabled() is False
    assert fresh_telemetry.setup_telemetry("test-service") is False


def test_disabled_when_sdk_disabled_flag(fresh_telemetry, monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    assert fresh_telemetry.telemetry_enabled() is False


def test_span_is_noop_when_disabled(fresh_telemetry):
    # No provider configured: span() must yield None and never raise.
    with fresh_telemetry.span("x.y", **{"attr": 1, "skip": None}) as s:
        assert s is None


def test_inject_context_empty_when_disabled(fresh_telemetry):
    assert fresh_telemetry.inject_context() == {}


def test_extract_context_none_when_disabled(fresh_telemetry):
    assert fresh_telemetry.extract_context({"traceparent": "abc"}) is None


def test_span_with_context_noop_when_disabled(fresh_telemetry):
    with fresh_telemetry.span_with_context("x.y", None) as s:
        assert s is None


# --- Live path: only when the OTLP HTTP exporter is actually installed -------

def _sdk_available() -> bool:
    try:
        import opentelemetry.exporter.otlp.proto.http.trace_exporter  # noqa: F401
        import opentelemetry.sdk.trace  # noqa: F401
        return True
    except ImportError:
        return False


sdk_required = pytest.mark.skipif(
    not _sdk_available(), reason="opentelemetry OTLP/HTTP exporter not installed"
)


@sdk_required
def test_setup_enables_and_propagates_context(fresh_telemetry, monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    # Configure the provider but export nothing, so the test never touches the
    # network / a real collector.
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "none")
    assert fresh_telemetry.setup_telemetry("test-service") is True
    assert fresh_telemetry.telemetry_enabled() is True

    # Inside an active span, inject_context must emit a W3C traceparent, and
    # extract_context must round-trip it back to a usable context.
    with fresh_telemetry.span("producer.op") as sp:
        assert sp is not None
        carrier = fresh_telemetry.inject_context()
        assert "traceparent" in carrier

    ctx = fresh_telemetry.extract_context(carrier)
    assert ctx is not None

    # A span started with the extracted context shares the producer's trace id.
    from opentelemetry import trace

    producer_trace_id = None
    with fresh_telemetry.span("producer.op2") as sp2:
        producer_trace_id = sp2.get_span_context().trace_id
        carrier2 = fresh_telemetry.inject_context()

    ctx2 = fresh_telemetry.extract_context(carrier2)
    with fresh_telemetry.span_with_context("consumer.op", ctx2) as consumer_span:
        assert consumer_span is not None
        assert consumer_span.get_span_context().trace_id == producer_trace_id
    # Silence unused-import lint if trace ends up unreferenced across versions.
    assert trace is not None
