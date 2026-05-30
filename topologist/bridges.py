from __future__ import annotations

import numpy as np
from typing import Any

from topologist.engine import Topologist

try:
    import torch  # type: ignore
    from torch_geometric.data import Data  # type: ignore
    PYTORCH_GEOMETRIC_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    torch = None  # type: ignore[assignment]
    Data = None  # type: ignore[assignment]
    PYTORCH_GEOMETRIC_AVAILABLE = False


class PyGBridge:
    """Bridge between Topologist and PyTorch Geometric."""

    def __init__(self, topology: Topologist) -> None:
        self.topology = topology

    @staticmethod
    def is_available() -> bool:
        return PYTORCH_GEOMETRIC_AVAILABLE

    def to_pyg_data(self) -> "Data":
        if not PYTORCH_GEOMETRIC_AVAILABLE or torch is None or Data is None:
            raise ImportError(
                "PyTorch Geometric is not installed."
                " Install with `pip install topologist[gnn]` to enable PyG bridging."
            )
        node_to_index = {node: idx for idx, node in enumerate(self.topology.graph.nodes())}
        x_vectors = []
        for node, data in self.topology.graph.nodes(data=True):
            hv = data.get("hv")
            if hv is None:
                hv = self.topology.hdc.get(f"node::{node}")
            x_vectors.append(hv.astype(np.float32))
        x_tensor = torch.from_numpy(np.stack(x_vectors, axis=0))

        edge_index = []
        edge_attr = []
        for source, target, data in self.topology.graph.edges(data=True):
            edge_index.append([node_to_index[source], node_to_index[target]])
            relation_hv = self.topology.hdc.get(f"relation::{data.get('relation', 'related_to')}")
            edge_attr.append(relation_hv.astype(np.float32))

        if edge_index:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr_tensor = torch.from_numpy(np.stack(edge_attr, axis=0))
        else:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
            edge_attr_tensor = torch.empty((0, self.topology.hdc.dim), dtype=torch.float32)

        return Data(x=x_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)

    @classmethod
    def from_pyg_data(cls, data: "Data") -> Topologist:
        raise NotImplementedError(
            "Deserializing PyG Data into a Topologist instance is not implemented yet."
        )
