"""Observability module for RAG system.

Provides:
- Distributed tracing (LangSmith/Arize integration)
- Metrics collection (accuracy, latency, cost)
- Artifact logging
- Dashboard definitions
"""

from .tracing import Tracer, Trace, TraceSpan
from .metrics import MetricsCollector, Metric, MetricType
from .artifacts import ArtifactLogger, Artifact, ArtifactType
from .dashboard import DashboardConfig, create_dashboard

__all__ = [
    "Tracer",
    "Trace",
    "TraceSpan",
    "MetricsCollector",
    "Metric",
    "MetricType",
    "ArtifactLogger",
    "Artifact",
    "ArtifactType",
    "DashboardConfig",
    "create_dashboard",
]
