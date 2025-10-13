"""Metrics collection utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from statistics import mean
from typing import Dict, List, Optional


class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class Metric:
    name: str
    metric_type: MetricType
    values: List[float] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)

    def record(self, value: float) -> None:
        self.values.append(value)

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def total(self) -> float:
        return sum(self.values)

    @property
    def average(self) -> Optional[float]:
        if not self.values:
            return None
        return mean(self.values)


class MetricsCollector:
    """In-memory metrics collector."""

    def __init__(self) -> None:
        self._metrics: Dict[str, Metric] = {}

    def _get_or_create_metric(self, name: str, metric_type: MetricType, tags: Optional[Dict[str, str]]) -> Metric:
        metric = self._metrics.get(name)
        if metric is None:
            metric = Metric(name=name, metric_type=metric_type, tags=tags or {})
            self._metrics[name] = metric
        return metric

    def increment(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        metric = self._get_or_create_metric(name, MetricType.COUNTER, tags)
        metric.record(value)

    def observe(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        metric = self._get_or_create_metric(name, MetricType.HISTOGRAM, tags)
        metric.record(value)

    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        metric = self._get_or_create_metric(name, MetricType.GAUGE, tags)
        metric.values = [value]

    def get_metric(self, name: str) -> Optional[Metric]:
        return self._metrics.get(name)

    def list_metrics(self) -> List[Metric]:
        return list(self._metrics.values())

    def clear(self) -> None:
        self._metrics.clear()
