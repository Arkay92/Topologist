from pathlib import Path

import pytest

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


def test_edge_provenance_and_reinforcement() -> None:
    topo = Topologist()
    created = topo.add_edge(
        "A",
        "supports",
        "B",
        confidence=0.6,
        source_type="sensor",
        evidence=["event-1"],
        trust_score=0.7,
    )
    reinforced = topo.add_edge(
        "A",
        "supports",
        "B",
        confidence=0.8,
        source_type="sensor",
        evidence=["event-2"],
        trust_score=0.9,
    )

    explanation = topo.explain_edge("A", "supports", "B")

    assert created is True
    assert reinforced is False
    assert explanation is not None
    assert explanation["source_type"] == "sensor"
    assert explanation["confidence"] == pytest.approx(0.8)
    assert explanation["trust_score"] == pytest.approx(0.9)
    assert explanation["reinforcement_count"] == 2
    assert explanation["evidence"] == ["event-1", "event-2"]
    assert explanation["derived"] is False


def test_explain_inferred_edge_contains_reasoning_trace() -> None:
    topo = Topologist()
    topo.add_edge("A", "causes", "B", confidence=0.9)
    topo.add_edge("B", "causes", "C", confidence=0.9)
    rule = ReasoningRule(
        relation_a="causes",
        relation_b="causes",
        inferred_relation="indirectly_causes",
        min_confidence=0.5,
    )

    assert topo.apply_rule(rule) == 1
    explanation = topo.explain_edge("A", "indirectly_causes", "C")

    assert explanation is not None
    assert explanation["derived"] is True
    assert explanation["source_type"] == "rule"
    assert explanation["confidence"] == pytest.approx(0.81)
    assert explanation["rule"]["inferred_relation"] == "indirectly_causes"
    assert explanation["evidence_path"] == [
        ["A", "causes", "B"],
        ["B", "causes", "C"],
    ]


def test_explain_edge_returns_none_for_missing_edge() -> None:
    topo = Topologist()

    assert topo.explain_edge("A", "missing", "B") is None


def test_detect_contradictions_reports_conflicting_edges_with_provenance() -> None:
    topo = Topologist()
    topo.add_edge(
        "service_api",
        "is_safe",
        "public_endpoint",
        confidence=0.9,
        source_type="user",
        evidence=["manual-review"],
    )
    topo.add_edge(
        "service_api",
        "is_risky",
        "public_endpoint",
        confidence=0.7,
        source_type="sensor",
        evidence=["scanner"],
    )

    contradictions = topo.detect_contradictions()

    assert len(contradictions) == 1
    report = contradictions[0]
    assert report["source"] == "service_api"
    assert report["target"] == "public_endpoint"
    assert set(report["relations"]) == {"is_safe", "is_risky"}
    assert report["severity"] == pytest.approx(0.7)
    assert report["left"]["evidence"] or report["right"]["evidence"]


def test_revise_belief_decays_old_edge_and_records_contradiction() -> None:
    topo = Topologist()
    topo.add_edge("service_api", "is_safe", "public_endpoint", confidence=0.8)

    result = topo.revise_belief(
        old=("service_api", "is_safe", "public_endpoint"),
        new=("service_api", "is_risky", "public_endpoint"),
        evidence="scanner finding",
        new_confidence=0.9,
    )

    old_explanation = topo.explain_edge("service_api", "is_safe", "public_endpoint")
    new_explanation = topo.explain_edge("service_api", "is_risky", "public_endpoint")
    contradiction = topo.explain_edge("service_api", "contradicts", "public_endpoint")

    assert old_explanation["confidence"] == pytest.approx(0.4)
    assert "scanner finding" in old_explanation["evidence"]
    assert new_explanation["confidence"] == pytest.approx(0.9)
    assert new_explanation["metadata"]["revision_of"] == [
        "service_api",
        "is_safe",
        "public_endpoint",
    ]
    assert contradiction is not None
    assert contradiction["metadata"]["revised_relation"] == "is_safe"
    assert result["contradictions"]


def test_revise_belief_rejects_missing_old_edge() -> None:
    topo = Topologist()

    with pytest.raises(ValueError, match="Cannot revise missing edge"):
        topo.revise_belief(
            old=("A", "is_safe", "B"),
            new=("A", "is_risky", "B"),
            evidence="scanner finding",
        )


def test_save_load_roundtrip(tmp_path: Path) -> None:
    topo = Topologist()
    topo.add_edge("HDC", "models", "Memory", source_type="user", evidence=["readme"])
    topo.update_global_state(take_snapshot=True)
    path = tmp_path / "topology.json"
    topo.save(path)
    loaded = Topologist.load(path)
    assert loaded.graph.number_of_nodes() == topo.graph.number_of_nodes()
    assert loaded.graph.number_of_edges() == topo.graph.number_of_edges()
    assert loaded.explain_edge("HDC", "models", "Memory")["source_type"] == "user"


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


def test_confidence_decay_reduces_confidence_values() -> None:
    topo = Topologist()
    topo.add_edge("A", "rel", "B", confidence=1.0)
    before = topo.neighbors("A")[0]["confidence"]
    assert before == 1.0
    topo.decay_confidence()
    after = topo.neighbors("A")[0]["confidence"]
    assert after < 1.0
    assert after == pytest.approx(0.98)


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
