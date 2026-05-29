from __future__ import annotations

import csv
import json
from pathlib import Path

from topologist.io import load_edges_csv, save_edges_csv
from topologist.models import EdgeRecord


def test_load_edges_csv_parses_metadata(tmp_path: Path) -> None:
    path = tmp_path / "edges.csv"
    rows = [
        {
            "source": "A",
            "relation": "causes",
            "target": "B",
            "weight": "0.8",
            "confidence": "0.9",
            "metadata_json": json.dumps({"source": "sensor"}),
        }
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    edges = load_edges_csv(path)
    assert len(edges) == 1
    assert edges[0].source == "A"
    assert edges[0].relation == "causes"
    assert edges[0].weight == 0.8
    assert edges[0].confidence == 0.9
    assert edges[0].metadata == {"source": "sensor"}


def test_save_edges_csv_writes_json_metadata(tmp_path: Path) -> None:
    path = tmp_path / "edges.csv"
    edge = EdgeRecord(
        source="A",
        relation="connects_to",
        target="B",
        weight=0.5,
        confidence=0.75,
        metadata={"type": "test"},
    )
    save_edges_csv(path, [edge])

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == 1
    assert json.loads(rows[0]["metadata_json"]) == edge.metadata
