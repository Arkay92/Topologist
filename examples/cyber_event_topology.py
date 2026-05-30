from __future__ import annotations

from pathlib import Path
from typing import Any

from topologist import MultiHopRule, Topologist, TopologistConfig
from topologist.visualization import export_mermaid


CYBER_EVENTS: list[dict[str, Any]] = [
    {
        "source": "10.0.3.42",
        "relation": "connects_to",
        "target": "ssh",
        "confidence": 0.92,
        "metadata": {"sensor": "netflow", "port": 22},
    },
    {
        "source": "ssh",
        "relation": "exposes",
        "target": "CVE-2025-1234",
        "confidence": 0.86,
        "metadata": {"scanner": "asset_inventory"},
    },
    {
        "source": "CVE-2025-1234",
        "relation": "enables",
        "target": "privilege_escalation",
        "confidence": 0.81,
        "metadata": {"evidence_source": "threat_intel"},
    },
    {
        "source": "10.0.3.42",
        "relation": "authenticates_to",
        "target": "domain_controller",
        "confidence": 0.67,
        "metadata": {"sensor": "edr"},
    },
    {
        "source": "unknown_external_host",
        "relation": "administers",
        "target": "domain_controller",
        "confidence": 0.44,
        "metadata": {"sensor": "edr", "severity": "suspicious"},
    },
]


def build_cyber_topology(output_dir: str | Path = ".") -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    topology = Topologist(TopologistConfig(dim=2048, seed=7, max_snapshots=16))
    escalation_rule = MultiHopRule(
        relation_sequence=["connects_to", "exposes", "enables"],
        inferred_relation="may_escalate_via",
        min_confidence=0.55,
        metadata={"scenario": "cyber_event_topology"},
    )

    drift_trace: list[float] = []
    inferred_total = 0
    for event in CYBER_EVENTS:
        topology.ingest_event(event)
        inferred_total += topology.apply_multi_hop_rule(escalation_rule)
        topology.update_global_state(take_snapshot=True)
        drift_trace.append(topology.topology_drift())

    suspicious_relation = (
        "unknown_external_host",
        "administers",
        "domain_controller",
    )
    anomaly_score = topology.relation_anomaly_score(*suspicious_relation)

    json_path = output_path / "cyber_event_topology.json"
    mermaid_path = output_path / "cyber_event_topology.mmd"
    topology.save(json_path)
    export_mermaid(topology, mermaid_path)

    return {
        "topology": topology,
        "inferred_edges": inferred_total,
        "anomaly_score": anomaly_score,
        "drift_trace": drift_trace,
        "json_path": json_path,
        "mermaid_path": mermaid_path,
    }


def main() -> None:
    result = build_cyber_topology()
    topology = result["topology"]

    print("Cyber event topology")
    print(f"Nodes: {topology.graph.number_of_nodes()}")
    print(f"Edges: {topology.graph.number_of_edges()}")
    print(f"Inferred edges: {result['inferred_edges']}")
    print(f"Suspicious relation anomaly score: {result['anomaly_score']:.4f}")
    print(f"Latest drift: {result['drift_trace'][-1]:.4f}")
    print(f"Mermaid export: {result['mermaid_path']}")
    print(f"JSON export: {result['json_path']}")


if __name__ == "__main__":
    main()
