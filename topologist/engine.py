from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import networkx as nx
import numpy as np

from topologist.config import TopologistConfig
from topologist.exceptions import NodeNotFoundError, PersistenceError
from topologist.hdc import BipolarVector, HyperVectorSpace
from topologist.models import EdgeRecord, NodeRecord, ReasoningRule

logger = logging.getLogger(__name__)


class Topologist:
    """Production-oriented hyperdimensional neuro-symbolic topology engine."""

    def __init__(self, config: TopologistConfig | None = None) -> None:
        self.config = config or TopologistConfig()
        self.hdc = HyperVectorSpace(dim=self.config.dim, seed=self.config.seed)
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.global_state: BipolarVector | None = None
        self.snapshots: list[BipolarVector] = []

    # -----------------------------
    # Node and edge mutation
    # -----------------------------

    def add_node(self, name: str, kind: str = "concept", **metadata: Any) -> None:
        record = NodeRecord(name=name, kind=kind, metadata=metadata)
        name_hv = self.hdc.get(f"node::{record.name}")
        kind_hv = self.hdc.get(f"kind::{record.kind}")
        hv = self.hdc.bind(kind_hv, name_hv)
        self.graph.add_node(
            record.name,
            kind=record.kind,
            metadata=record.metadata,
            name_hv=name_hv,
            kind_hv=kind_hv,
            hv=hv,
        )
        self.global_state = None
        logger.debug("Added node %s kind=%s", record.name, record.kind)

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        weight: float | None = None,
        confidence: float | None = None,
        **metadata: Any,
    ) -> bool:
        record = EdgeRecord(
            source=source,
            relation=relation,
            target=target,
            weight=self.config.default_weight if weight is None else weight,
            confidence=self.config.default_confidence if confidence is None else confidence,
            metadata=metadata,
        )
        if record.source not in self.graph:
            self.add_node(record.source)
        if record.target not in self.graph:
            self.add_node(record.target)
        if self._has_edge(
            record.source,
            record.relation,
            record.target,
            record.weight,
            record.confidence,
            record.metadata,
        ):
            self._update_edge_confidence(
                record.source,
                record.target,
                record.relation,
                record.confidence,
            )
            logger.debug(
                "Updated confidence for edge %s --%s--> %s to %.3f",
                record.source,
                record.relation,
                record.target,
                record.confidence,
            )
            return False
        hv = self.hdc.encode_relation(record.source, record.relation, record.target)
        self.graph.add_edge(
            record.source,
            record.target,
            relation=record.relation,
            weight=record.weight,
            confidence=record.confidence,
            metadata=record.metadata,
            hv=hv,
        )
        self.global_state = None
        logger.debug(
            "Added edge %s --%s--> %s confidence=%.3f",
            record.source,
            record.relation,
            record.target,
            record.confidence,
        )
        return True

    def remove_node(self, name: str) -> None:
        if name not in self.graph:
            raise NodeNotFoundError(f"node not found: {name}")
        self.graph.remove_node(name)
        self.global_state = None

    def _has_edge(
        self,
        source: str,
        relation: str,
        target: str,
        weight: float,
        confidence: float,
        metadata: dict[str, Any],
    ) -> bool:
        """Check if an edge exists by symbolic key (source, relation, target).
        For inferred edges, also match rule identity if present.
        """
        edge_data = self.graph.get_edge_data(source, target, default={})
        for data in edge_data.values():
            if data.get("relation") != relation:
                continue
            if not metadata and not data.get("metadata"):
                return True
            if metadata and data.get("metadata"):
                existing_rule = data.get("metadata", {}).get("rule")
                new_rule = metadata.get("rule")
                if existing_rule == new_rule:
                    return True
        return False
    
    def _update_edge_confidence(
        self,
        source: str,
        target: str,
        relation: str,
        new_confidence: float,
    ) -> None:
        """Update an existing edge's confidence to the max of current and new."""
        edge_data = self.graph.get_edge_data(source, target, default={})
        for data in edge_data.values():
            if data.get("relation") == relation:
                data["confidence"] = max(float(data.get("confidence", 1.0)), new_confidence)
                self.global_state = None
                return

    # -----------------------------
    # HDC state and snapshots
    # -----------------------------

    def update_global_state(self, take_snapshot: bool = False) -> BipolarVector:
        vectors: list[BipolarVector] = []
        for _, data in self.graph.nodes(data=True):
            # include typed node vector (already kind-bound)
            vectors.append(data["hv"])
        for _, _, data in self.graph.edges(data=True):
            # incorporate confidence into the edge vector so confidence decay
            # affects the global topology memory. We quantize confidence to
            # a single decimal to reduce item-memory growth.
            edge_hv = data.get("hv")
            try:
                conf = float(data.get("confidence", 1.0))
            except Exception:
                conf = 1.0
            # encode confidence as a continuous permuted vector
            conf_hv = self.hdc.encode_confidence(conf)
            vectors.append(self.hdc.bind(edge_hv, conf_hv))
        self.global_state = self.hdc.bundle(vectors)
        if take_snapshot:
            self.snapshots.append(self.global_state.copy())
            self.snapshots = self.snapshots[-self.config.max_snapshots :]
        return self.global_state

    def state_similarity(self, other: "Topologist") -> float:
        left = self.global_state if self.global_state is not None else self.update_global_state()
        right = other.global_state if other.global_state is not None else other.update_global_state()
        return self.hdc.similarity(left, right)

    def topology_drift(self) -> float:
        if len(self.snapshots) < 2:
            return 0.0
        return 1.0 - self.hdc.similarity(self.snapshots[-2], self.snapshots[-1])

    # -----------------------------
    # Retrieval and reasoning
    # -----------------------------

    def nearest_nodes(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        query_name_hv = self.hdc.get(f"node::{query}")
        scores = []
        for node, data in self.graph.nodes(data=True):
            name_hv = data.get("name_hv", data["hv"])
            typed_hv = data.get("hv", name_hv)
            score = max(
                self.hdc.similarity(query_name_hv, name_hv),
                self.hdc.similarity(query_name_hv, typed_hv),
            )
            if score >= self.config.similarity_floor:
                scores.append((node, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]

    def neighbors(self, node: str) -> list[dict[str, Any]]:
        if node not in self.graph:
            raise NodeNotFoundError(f"node not found: {node}")
        return [
            {
                "source": source,
                "relation": data["relation"],
                "target": target,
                "weight": data.get("weight", 1.0),
                "confidence": data.get("confidence", 1.0),
                "metadata": data.get("metadata", {}),
            }
            for source, target, data in self.graph.out_edges(node, data=True)
        ]

    def apply_rule(self, rule: ReasoningRule) -> int:
        proposed: list[tuple[str, str, str, float]] = []
        for a, b, data_ab in self.graph.edges(data=True):
            if data_ab.get("relation") != rule.relation_a:
                continue
            for _, c, data_bc in self.graph.out_edges(b, data=True):
                if data_bc.get("relation") != rule.relation_b:
                    continue
                confidence = data_ab.get("confidence", 1.0) * data_bc.get("confidence", 1.0)
                if confidence >= rule.min_confidence:
                    proposed.append((a, rule.inferred_relation, c, confidence))
        created = 0
        for source, relation, target, confidence in proposed:
            if self.add_edge(
                source,
                relation,
                target,
                confidence=confidence,
                inferred=True,
                rule=rule.model_dump(),
            ):
                created += 1
        return created

    def apply_rules(self, rules: list[ReasoningRule]) -> int:
        return sum(self.apply_rule(rule) for rule in rules)

    def ingest_event(self, event: dict[str, Any]) -> bool:
        """Ingest a generic topology event into the engine."""
        source = str(event.get("source", "")).strip()
        relation = str(event.get("relation", "")).strip()
        target = str(event.get("target", "")).strip()
        if not source or not relation or not target:
            raise ValueError("Event must include source, relation, and target.")
        weight = float(event.get("weight", self.config.default_weight))
        confidence = float(event.get("confidence", self.config.default_confidence))
        metadata = event.get("metadata", {}) or {}
        return self.add_edge(source, relation, target, weight=weight, confidence=confidence, **metadata)

    def apply_multi_hop_rule(self, rule: "MultiHopRule") -> int:
        created = 0
        proposals: list[tuple[str, str, float]] = []

        def traverse(current_node: str, relation_index: int, confidence: float, start_node: str) -> None:
            if relation_index >= len(rule.relation_sequence):
                if confidence >= rule.min_confidence:
                    proposals.append((start_node, current_node, confidence))
                return
            relation = rule.relation_sequence[relation_index]
            for _, next_node, data in self.graph.out_edges(current_node, data=True):
                if data.get("relation") != relation:
                    continue
                next_confidence = confidence * float(data.get("confidence", 1.0))
                traverse(next_node, relation_index + 1, next_confidence, start_node)

        for start_node in list(self.graph.nodes()):
            traverse(start_node, 0, 1.0, start_node)

        for source, target, confidence in proposals:
            if self.add_edge(
                source,
                rule.inferred_relation,
                target,
                confidence=confidence,
                inferred=True,
                rule=rule.model_dump(),
            ):
                created += 1

        return created

    def apply_dsl_rule(self, expression: str) -> int:
        from topologist.dsl import RuleDSL

        rule = RuleDSL.parse(expression)
        return self.apply_multi_hop_rule(rule)

    def summarize_topology(self) -> str:
        nodes = sorted(self.graph.nodes())
        edges = [
            f"{source}-[{data.get('relation', 'related_to')}]->{target}"
            for source, target, data in self.graph.edges(data=True)
        ]
        return f"nodes: {', '.join(nodes)}\nedges: {', '.join(edges)}"

    def save_to_sqlite(self, path: str | Path) -> None:
        from topologist.persistence import SQLitePersistenceAdapter

        adapter = SQLitePersistenceAdapter(path)
        adapter.save(self)
        adapter.close()

    @classmethod
    def load_from_sqlite(cls, path: str | Path) -> "Topologist":
        from topologist.persistence import SQLitePersistenceAdapter

        adapter = SQLitePersistenceAdapter(path)
        topology = adapter.load()
        adapter.close()
        return topology

    # -----------------------------
    # Topological analytics
    # -----------------------------

    def centrality(self) -> dict[str, float]:
        simple = nx.DiGraph()
        for source, target, data in self.graph.edges(data=True):
            simple.add_edge(source, target, weight=data.get("weight", 1.0))
        if simple.number_of_nodes() == 0:
            return {}
        return cast(dict[str, float], nx.pagerank(simple, weight="weight"))

    def communities(self) -> list[list[str]]:
        if self.graph.number_of_nodes() == 0:
            return []
        undirected = self.graph.to_undirected()
        groups = nx.algorithms.community.greedy_modularity_communities(undirected)
        return [sorted(group) for group in groups]

    def shortest_path(self, source: str, target: str) -> list[str] | None:
        try:
            return cast(list[str], nx.shortest_path(self.graph, source=source, target=target))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def relation_anomaly_score(self, source: str, relation: str, target: str) -> float:
        """Compute an anomaly score for a candidate relation using unbinding.

        Rather than comparing the bound relation (source*relation*target) to
        individual components, we unbind the candidate and recover the
        relation signal. This is more mathematically sound in VSA/HDC.

        The recovered relation is compared against:
        1. The expected relation hypervector
        2. Local topology bundles (outgoing from source, incoming to target)

        Returns anomaly in range [0, 1], where 1 is most anomalous.
        """
        candidate = self.hdc.encode_relation(source, relation, target)
        source_hv = self.hdc.get(f"node::{source}")
        target_hv = self.hdc.get(f"node::{target}")
        expected_relation_hv = self.hdc.get(f"relation::{relation}")

        # unbind to recover relation signal: bind(bind(candidate, source), target)
        recovered_relation = self.hdc.bind(self.hdc.bind(candidate, source_hv), target_hv)
        relation_score = self.hdc.similarity(recovered_relation, expected_relation_hv)

        # compare against local topology
        local_scores: list[float] = []
        for _, _, data in self.graph.out_edges(source, data=True):
            edge_hv = data.get("hv")
            if edge_hv is not None:
                local_scores.append(self.hdc.similarity(candidate, edge_hv))

        for _, _, data in self.graph.in_edges(target, data=True):
            edge_hv = data.get("hv")
            if edge_hv is not None:
                local_scores.append(self.hdc.similarity(candidate, edge_hv))

        best_local = max(local_scores, default=0.0)
        best_score = max(relation_score, best_local)

        # anomaly is inverse of best match, clamped to [0, 1]
        return max(0.0, min(1.0, 1.0 - best_score))

    def is_anomalous_relation(self, source: str, relation: str, target: str) -> bool:
        return self.relation_anomaly_score(source, relation, target) >= self.config.anomaly_threshold

    def decay_confidence(self) -> None:
        """Apply confidence decay to all edges, useful for stale knowledge."""
        for _, _, data in self.graph.edges(data=True):
            data["confidence"] = max(0.0, float(data.get("confidence", 1.0)) * self.config.decay_rate)
        self.global_state = None

    # -----------------------------
    # Serialization
    # -----------------------------

    def to_dict(self, include_hdc_memory: bool = True) -> dict[str, Any]:
        nodes = [
            NodeRecord(
                name=node,
                kind=data.get("kind", "concept"),
                metadata=data.get("metadata", {}),
            ).model_dump()
            for node, data in self.graph.nodes(data=True)
        ]
        edges = [
            EdgeRecord(
                source=source,
                relation=data.get("relation", "related_to"),
                target=target,
                weight=float(data.get("weight", 1.0)),
                confidence=float(data.get("confidence", 1.0)),
                metadata=data.get("metadata", {}),
            ).model_dump()
            for source, target, data in self.graph.edges(data=True)
        ]
        payload: dict[str, Any] = {
            "config": self.config.model_dump(),
            "nodes": nodes,
            "edges": edges,
            "snapshots": [snapshot.tolist() for snapshot in self.snapshots],
        }
        if include_hdc_memory:
            payload["hdc"] = self.hdc.to_jsonable()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Topologist":
        config = TopologistConfig(**payload.get("config", {}))
        engine = cls(config=config)
        if "hdc" in payload:
            engine.hdc = HyperVectorSpace.from_jsonable(payload["hdc"])
        for node in payload.get("nodes", []):
            node_record = NodeRecord(**node)
            engine.add_node(node_record.name, node_record.kind, **node_record.metadata)
        for edge in payload.get("edges", []):
            edge_record = EdgeRecord(**edge)
            engine.add_edge(
                edge_record.source,
                edge_record.relation,
                edge_record.target,
                weight=edge_record.weight,
                confidence=edge_record.confidence,
                **edge_record.metadata,
            )
        engine.snapshots = [np.asarray(snapshot, dtype=np.int8) for snapshot in payload.get("snapshots", [])]
        engine.update_global_state()
        return engine

    def save(self, path: str | Path) -> None:
        try:
            Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover
            raise PersistenceError(f"failed to save topology: {exc}") from exc

    @classmethod
    def load(cls, path: str | Path) -> "Topologist":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            return cls.from_dict(payload)
        except Exception as exc:  # pragma: no cover
            raise PersistenceError(f"failed to load topology: {exc}") from exc

    def export_graphml(self, path: str | Path) -> None:
        """Export symbolic topology to GraphML without raw hypervectors."""
        export_graph = nx.MultiDiGraph()
        for node, data in self.graph.nodes(data=True):
            export_graph.add_node(
                node,
                kind=data.get("kind", "concept"),
                metadata=json.dumps(data.get("metadata", {})),
            )
        for source, target, data in self.graph.edges(data=True):
            export_graph.add_edge(
                source,
                target,
                relation=data.get("relation", "related_to"),
                weight=data.get("weight", 1.0),
                confidence=data.get("confidence", 1.0),
                metadata=json.dumps(data.get("metadata", {})),
            )
        nx.write_graphml(export_graph, path)
