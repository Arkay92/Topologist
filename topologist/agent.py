from __future__ import annotations

import json
from abc import ABC, abstractmethod

from topologist.engine import Topologist


class AgentMemoryAdapter(ABC):
    """Abstract adapter for agent memory integration."""

    @abstractmethod
    def persist(self, topology: Topologist) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, prompt: str) -> str:
        raise NotImplementedError


class LocalLLMAgentMemoryAdapter(AgentMemoryAdapter):
    def __init__(self) -> None:
        self.memory: list[str] = []

    def persist(self, topology: Topologist) -> None:
        self.memory.append(self._serialize_topology(topology))

    def query(self, prompt: str) -> str:
        matches = [entry for entry in self.memory if prompt.lower() in entry.lower()]
        if not matches:
            return "No relevant memory found."
        return matches[-1]

    def _serialize_topology(self, topology: Topologist) -> str:
        nodes = sorted(topology.graph.nodes())
        edges = [
            f"{source}-[{data.get('relation', 'related_to')}]->{target}"
            for source, target, data in topology.graph.edges(data=True)
        ]
        return json.dumps({"nodes": nodes, "edges": edges})


class ClaudeCodeAdapter(AgentMemoryAdapter):
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self.memory: list[str] = []

    def persist(self, topology: Topologist) -> None:
        self.memory.append(self._serialize_topology(topology))

    def query(self, prompt: str) -> str:
        if not self.api_key:
            return "Claude Code adapter configured without an API key."
        return f"Claude Code query result placeholder for prompt: {prompt}"

    def _serialize_topology(self, topology: Topologist) -> str:
        return LocalLLMAgentMemoryAdapter()._serialize_topology(topology)


class OpenClawAdapter(AgentMemoryAdapter):
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self.memory: list[str] = []

    def persist(self, topology: Topologist) -> None:
        self.memory.append(self._serialize_topology(topology))

    def query(self, prompt: str) -> str:
        if not self.api_key:
            return "OpenClaw adapter configured without an API key."
        return f"OpenClaw query result placeholder for prompt: {prompt}"

    def _serialize_topology(self, topology: Topologist) -> str:
        return LocalLLMAgentMemoryAdapter()._serialize_topology(topology)
