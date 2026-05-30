from __future__ import annotations

from contextlib import contextmanager
from typing import Any, ContextManager

try:
    from opentelemetry import trace  # type: ignore
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore
    OPENTELEMETRY_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    trace = None  # type: ignore[assignment]
    TracerProvider = None  # type: ignore[assignment]
    OPENTELEMETRY_AVAILABLE = False


class OpenTelemetryTracer:
    """Optional OpenTelemetry tracer helper for Topologist."""

    def __init__(self, service_name: str = "topologist") -> None:
        self.service_name = service_name
        self.tracer = None
        if OPENTELEMETRY_AVAILABLE and trace is not None and TracerProvider is not None:
            provider = TracerProvider()
            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer(service_name)

    @staticmethod
    def is_available() -> bool:
        return OPENTELEMETRY_AVAILABLE

    @contextmanager
    def trace_operation(self, name: str) -> ContextManager[Any]:
        if self.tracer is None:
            yield
            return
        with self.tracer.start_as_current_span(name):
            yield
