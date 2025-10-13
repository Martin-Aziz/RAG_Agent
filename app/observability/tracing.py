"""Lightweight tracing utilities for the RAG system."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional


@dataclass
class TraceSpan:
    """Represents a span within a trace."""

    name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def finish(self) -> None:
        if self.end_time is None:
            self.end_time = time.time()

    @property
    def duration(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return self.end_time - self.start_time


@dataclass
class Trace:
    """Captures a collection of spans for a single operation."""

    name: str
    spans: List[TraceSpan] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)

    def add_span(self, span: TraceSpan) -> None:
        self.spans.append(span)

    @contextmanager
    def span(self, name: str, metadata: Optional[Dict[str, str]] = None) -> Iterator[TraceSpan]:
        span = TraceSpan(name=name, metadata=metadata or {})
        try:
            yield span
        finally:
            span.finish()
            self.add_span(span)


class Tracer:
    """Simple in-memory tracer for collecting traces and spans."""

    def __init__(self) -> None:
        self._traces: List[Trace] = []

    def start_trace(self, name: str, metadata: Optional[Dict[str, str]] = None) -> Trace:
        trace = Trace(name=name, metadata=metadata or {})
        self._traces.append(trace)
        return trace

    def get_traces(self) -> List[Trace]:
        return list(self._traces)

    def clear(self) -> None:
        self._traces.clear()
