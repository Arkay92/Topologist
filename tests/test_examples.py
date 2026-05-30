from __future__ import annotations

from pathlib import Path

from examples.cyber_event_topology import build_cyber_topology


def test_cyber_event_topology_example_exports_outputs(tmp_path: Path) -> None:
    result = build_cyber_topology(tmp_path)
    topology = result["topology"]

    assert result["inferred_edges"] == 1
    assert 0.0 <= result["anomaly_score"] <= 1.0
    assert len(result["drift_trace"]) == 5
    assert topology.graph.has_edge("10.0.3.42", "privilege_escalation")
    assert result["json_path"].exists()
    assert result["mermaid_path"].exists()
    assert "may_escalate_via" in result["mermaid_path"].read_text(encoding="utf-8")
