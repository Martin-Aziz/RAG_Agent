"""Dashboard configuration utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .metrics import Metric


@dataclass
class DashboardConfig:
    title: str
    description: Optional[str] = None
    refresh_interval_seconds: int = 30
    tracked_metrics: List[str] = field(default_factory=list)
    additional_settings: Dict[str, str] = field(default_factory=dict)

    def add_metric(self, metric_name: str) -> None:
        if metric_name not in self.tracked_metrics:
            self.tracked_metrics.append(metric_name)

    def to_dict(self) -> Dict[str, object]:
        return {
            "title": self.title,
            "description": self.description,
            "refresh_interval_seconds": self.refresh_interval_seconds,
            "tracked_metrics": list(self.tracked_metrics),
            "additional_settings": dict(self.additional_settings),
        }


def create_dashboard(title: str, metrics: List[Metric], description: Optional[str] = None) -> DashboardConfig:
    config = DashboardConfig(title=title, description=description)
    for metric in metrics:
        config.add_metric(metric.name)
    return config
