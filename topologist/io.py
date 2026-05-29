from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from topologist.engine import Topologist
from topologist.models import EdgeRecord, NodeRecord


def load_edges_csv(path: str | Path) -> list[EdgeRecord]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        edges: list[EdgeRecord] = []
        for row in reader:
            metadata = {}
            metadata_json = row.get("metadata_json")
            if metadata_json:
                metadata = json.loads(metadata_json)
            elif row.get("metadata"):
                raw_metadata = row["metadata"]
                try:
                    metadata = json.loads(raw_metadata)
                except json.JSONDecodeError:
                    metadata = {"metadata": raw_metadata}
            weight = float(row["weight"]) if row.get("weight") not in (None, "") else 1.0
            confidence = float(row["confidence"]) if row.get("confidence") not in (None, "") else 1.0
            edges.append(
                EdgeRecord(
                    source=row["source"],
                    relation=row["relation"],
                    target=row["target"],
                    weight=weight,
                    confidence=confidence,
                    metadata=metadata,
                )
            )
        return edges


def save_edges_csv(path: str | Path, edges: Iterable[EdgeRecord]) -> None:
    rows: list[dict[str, object]] = []
    for edge in edges:
        edge_data = edge.model_dump()
        edge_data["metadata_json"] = json.dumps(edge_data.pop("metadata"))
        rows.append(edge_data)
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_from_records(nodes: Iterable[NodeRecord], edges: Iterable[EdgeRecord]) -> Topologist:
    topology = Topologist()
    for node in nodes:
        topology.add_node(node.name, node.kind, **node.metadata)
    for edge in edges:
        topology.add_edge(
            edge.source,
            edge.relation,
            edge.target,
            weight=edge.weight,
            confidence=edge.confidence,
            **edge.metadata,
        )
    topology.update_global_state(take_snapshot=True)
    return topology
