"""
EcodiaOS - Observability Infrastructure

Structured logging, metrics collection, and tracing.
"""

from telemetry.logging import setup_logging
from telemetry.metrics import MetricCollector

__all__ = ["setup_logging", "MetricCollector"]
