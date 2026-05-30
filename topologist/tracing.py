from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

try:
    from opentelemetry import trace as _trace
    from opentelemetry.sdk.trace import TracerProvider as _TracerProvider

    trace: Any = _trace
    TracerProvider: Any = _TracerProvider
    OPENTELEMETRY_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    trace = None
    TracerProvider = None
    OPENTELEMETRY_AVAILABLE = False


class OpenTelemetryTracer:
    """Optional OpenTelemetry tracer helper for Topologist."""

    def __init__(self, service_name: str = "topologist") -> None:
        self.service_name = service_name
        self.tracer: Any = None
        if OPENTELEMETRY_AVAILABLE and trace is not None and TracerProvider is not None:
            provider = TracerProvider()
            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer(service_name)

    @staticmethod
    def is_available() -> bool:
        return OPENTELEMETRY_AVAILABLE

    @contextmanager
    def trace_operation(self, name: str) -> Iterator[None]:
        if self.tracer is None:
            yield
            return
        with self.tracer.start_as_current_span(name):
            yield
