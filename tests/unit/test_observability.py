from pathlib import Path

import pytest

from app.observability.artifacts import ArtifactLogger, ArtifactType
from app.observability.dashboard import create_dashboard
from app.observability.metrics import MetricsCollector
from app.observability.tracing import Tracer


def test_tracer_creates_span():
    tracer = Tracer()
    trace = tracer.start_trace("test")
    with trace.span("child") as span:
        assert span.name == "child"
    assert trace.spans[0].duration is not None


def test_metrics_collector_tracks_values():
    collector = MetricsCollector()
    collector.increment("requests")
    collector.observe("latency", 123.0)
    collector.gauge("queue_depth", 5)

    requests_metric = collector.get_metric("requests")
    latency_metric = collector.get_metric("latency")
    queue_metric = collector.get_metric("queue_depth")

    assert requests_metric and requests_metric.count == 1
    assert latency_metric and latency_metric.average == pytest.approx(123.0)
    assert queue_metric and queue_metric.values == [5]


def test_artifact_logger_persists(tmp_path: Path):
    logger = ArtifactLogger()
    artifact = logger.log_text("response", "hello world")
    saved_path = artifact.persist(tmp_path)

    assert saved_path.exists()
    assert saved_path.read_text() == "hello world"


def test_create_dashboard_includes_metrics():
    collector = MetricsCollector()
    collector.increment("requests")
    dashboard = create_dashboard("main", collector.list_metrics())

    assert "requests" in dashboard.tracked_metrics
    assert dashboard.to_dict()["title"] == "main"
