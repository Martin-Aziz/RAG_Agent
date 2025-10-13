import logging
import uuid
try:
    from prometheus_client import Counter, Histogram
except Exception:
    # lightweight fallback classes for environments without prometheus_client
    class _NoopMetric:
        def __init__(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

        def inc(self, amount=1):
            return None

        def observe(self, value):
            return None

    Counter = _NoopMetric
    Histogram = _NoopMetric
from typing import Dict


# Metrics
REQUEST_COUNTER = Counter("agenticrag_requests_total", "Total requests processed", ["mode"])
LATENCY_HIST = Histogram("agenticrag_request_latency_seconds", "Request latency seconds", ["mode"])
AGENT_COUNTER = Counter("agenticrag_agent_actions_total", "Number of agent actions executed", ["agent"])
LLM_CALLS = Counter("agenticrag_llm_calls_total", "Number of LLM/SLM calls", ["model"])


def get_logger(name: str = "agenticrag") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)s %(request_id)s %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def new_request_id() -> str:
    return uuid.uuid4().hex


def redact_pii(text: str) -> str:
    # naive PII redaction: redact emails and phone-like numbers
    import re
    text = re.sub(r"[\w.-]+@[\w.-]+", "[REDACTED_EMAIL]", text)
    text = re.sub(r"\+?\d[\d\-() ]{6,}\d", "[REDACTED_PHONE]", text)
    return text
