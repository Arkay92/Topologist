from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any

from topologist.engine import Topologist

try:
    from annoy import AnnoyIndex as _AnnoyIndex

    AnnoyIndex: Any = _AnnoyIndex
    ANNOY_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    AnnoyIndex = None
    ANNOY_AVAILABLE = False


class ApproximateNearestNeighbor:
    """Approximate nearest-neighbor search for Topologist item memories."""

    def __init__(self, topology: Topologist, use_annoy: bool = False, metric: str = "angular") -> None:
        self.topology = topology
        self.use_annoy = use_annoy and ANNOY_AVAILABLE
        self.metric = metric
        self.index: Any = None
        self.vector_store: dict[str, NDArray[Any]] = {}

    def build_index(self) -> None:
        self.vector_store = {
            node: data.get("hv", self.topology.hdc.get(f"node::{node}"))
            for node, data in self.topology.graph.nodes(data=True)
        }
        if self.use_annoy and AnnoyIndex is not None:
            index = AnnoyIndex(self.topology.hdc.dim, self.metric)
            for idx, (node, vector) in enumerate(self.vector_store.items()):
                index.add_item(idx, vector.astype(np.float32).tolist())
            index.build(10)
            self.index = index

    def query(self, query: str | NDArray[Any], top_k: int = 5) -> list[tuple[str, float]]:
        if isinstance(query, str):
            query_vector = self.topology.hdc.get(f"node::{query}")
        else:
            query_vector = query
        if not self.vector_store:
            self.build_index()
        if self.use_annoy and self.index is not None:
            ids = self.index.get_nns_by_vector(query_vector.astype(np.float32).tolist(), top_k, include_distances=True)
            node_names = list(self.vector_store.keys())
            return [
                (node_names[idx], 1.0 - float(dist) / 2.0)
                for idx, dist in zip(ids[0], ids[1])
            ]
        similarities: list[tuple[str, float]] = []
        for node, vector in self.vector_store.items():
            if vector is None:
                continue
            score = self.topology.hdc.similarity(query_vector, vector)
            similarities.append((node, score))
        return sorted(similarities, key=lambda item: item[1], reverse=True)[:top_k]

    def has_annoy_support(self) -> bool:
        return self.use_annoy and ANNOY_AVAILABLE
