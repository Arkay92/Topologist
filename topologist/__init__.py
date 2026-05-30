"""Topologist: hyperdimensional neuro-symbolic topology engine."""

from topologist.agent import ClaudeCodeAdapter, LocalLLMAgentMemoryAdapter, OpenClawAdapter
from topologist.ann import ApproximateNearestNeighbor
from topologist.bridges import PyGBridge
from topologist.config import TopologistConfig
from topologist.dsl import MultiHopRule, RuleDSL
from topologist.engine import Topologist
from topologist.hdc import HyperVectorSpace
from topologist.models import EdgeRecord, NodeRecord, ProvenanceRecord, ReasoningRule
from topologist.persistence import PostgresPersistenceAdapter, SQLitePersistenceAdapter
from topologist.service import create_app
from topologist.streaming import EventStreamAdapter, KafkaStreamAdapter, RedisStreamAdapter, WebSocketStreamAdapter
from topologist.tracing import OpenTelemetryTracer

__all__ = [
    "Topologist",
    "TopologistConfig",
    "HyperVectorSpace",
    "NodeRecord",
    "EdgeRecord",
    "ProvenanceRecord",
    "ReasoningRule",
    "MultiHopRule",
    "RuleDSL",
    "PyGBridge",
    "ApproximateNearestNeighbor",
    "SQLitePersistenceAdapter",
    "PostgresPersistenceAdapter",
    "EventStreamAdapter",
    "KafkaStreamAdapter",
    "RedisStreamAdapter",
    "WebSocketStreamAdapter",
    "OpenTelemetryTracer",
    "LocalLLMAgentMemoryAdapter",
    "ClaudeCodeAdapter",
    "OpenClawAdapter",
    "create_app",
]

try:
    from importlib.metadata import version

    __version__ = version("topologist")
except Exception:
    __version__ = "0.2.0"
