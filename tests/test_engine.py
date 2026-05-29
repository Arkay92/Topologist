from pathlib import Path

from topologist import Topologist
from topologist.models import ReasoningRule


def test_add_nodes_edges_and_neighbors() -> None:
    topo = Topologist()
    topo.add_edge("A", "causes", "B", confidence=0.8)
    neighbors = topo.neighbors("A")
    assert neighbors[0]["target"] == "B"
    assert neighbors[0]["relation"] == "causes"


def test_rule_inference() -> None:
    topo = Topologist()
    topo.add_edge("A", "causes", "B", confidence=0.9)
    topo.add_edge("B", "causes", "C", confidence=0.9)
    created = topo.apply_rule(
        ReasoningRule(
            relation_a="causes",
            relation_b="causes",
            inferred_relation="indirectly_causes",
            min_confidence=0.5,
        )
    )
    assert created == 1
    assert topo.shortest_path("A", "C") is not None


def test_save_load_roundtrip(tmp_path: Path) -> None:
    topo = Topologist()
    topo.add_edge("HDC", "models", "Memory")
    topo.update_global_state(take_snapshot=True)
    path = tmp_path / "topology.json"
    topo.save(path)
    loaded = Topologist.load(path)
    assert loaded.graph.number_of_nodes() == topo.graph.number_of_nodes()
    assert loaded.graph.number_of_edges() == topo.graph.number_of_edges()


def test_decay_confidence() -> None:
    topo = Topologist()
    topo.add_edge("A", "rel", "B", confidence=1.0)
    topo.decay_confidence()
    neighbor = topo.neighbors("A")[0]
    assert neighbor["confidence"] < 1.0


def test_nearest_nodes_prefers_name_match() -> None:
    topo = Topologist()
    topo.add_node("Memory", "concept")
    topo.add_node("Model", "concept")
    results = topo.nearest_nodes("Memory", top_k=2)
    assert results[0][0] == "Memory"


def test_anomaly_score_is_clamped() -> None:
    topo = Topologist()
    topo.add_node("A", "concept")
    topo.add_node("B", "concept")
    # anomaly score should be in [0, 1] even if similarity is negative
    score = topo.relation_anomaly_score("A", "any_relation", "B")
    assert 0.0 <= score <= 1.0


def test_inferred_edge_deduplication_with_confidence_update() -> None:
    topo = Topologist()
    topo.add_edge("A", "causes", "B", confidence=0.9)
    topo.add_edge("B", "causes", "C", confidence=0.9)
    rule = ReasoningRule(
        relation_a="causes",
        relation_b="causes",
        inferred_relation="indirectly_causes",
        min_confidence=0.5,
    )
    # first application
    created_1 = topo.apply_rule(rule)
    assert created_1 == 1
    edge_1 = list(topo.graph.get_edge_data("A", "C").values())[0]
    conf_1 = edge_1["confidence"]
    # second application with same rule (no new inferred edge)
    created_2 = topo.apply_rule(rule)
    assert created_2 == 0
    # confidence should remain unchanged since it's the same edge
    edge_2 = list(topo.graph.get_edge_data("A", "C").values())[0]
    assert edge_2["confidence"] == conf_1


def test_confidence_decay_changes_global_state() -> None:
    topo = Topologist()
    topo.add_edge("A", "rel", "B", confidence=1.0)
    topo.update_global_state(take_snapshot=True)
    # snapshot[0] exists
    topo.decay_confidence()
    topo.update_global_state(take_snapshot=True)
    # drift should be non-zero after confidence decay
    assert topo.topology_drift() > 0.0


def test_rule_inference_is_idempotent() -> None:
    topo = Topologist()
    topo.add_edge("A", "causes", "B", confidence=0.9)
    topo.add_edge("B", "causes", "C", confidence=0.9)
    created_first = topo.apply_rule(
        ReasoningRule(
            relation_a="causes",
            relation_b="causes",
            inferred_relation="indirectly_causes",
            min_confidence=0.5,
        )
    )
    created_second = topo.apply_rule(
        ReasoningRule(
            relation_a="causes",
            relation_b="causes",
            inferred_relation="indirectly_causes",
            min_confidence=0.5,
        )
    )
    assert created_first == 1
    assert created_second == 0
    inferred_edges = [
        data
        for _, _, data in topo.graph.edges(data=True)
        if data.get("relation") == "indirectly_causes"
    ]
    assert len(inferred_edges) == 1
