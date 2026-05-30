from pathlib import Path

import pytest

from topologist import Topologist, TopologistConfig
from topologist.agent import ClaudeCodeAdapter, LocalLLMAgentMemoryAdapter, OpenClawAdapter
from topologist.ann import ApproximateNearestNeighbor
from topologist.dsl import MultiHopRule, RuleDSL
from topologist.persistence import SQLitePersistenceAdapter


def test_ingest_event_creates_edge() -> None:
    topo = Topologist()
    event = {"source": "A", "relation": "rel", "target": "B", "confidence": 0.75}
    assert topo.ingest_event(event)
    assert topo.graph.has_edge("A", "B")


def test_parse_multi_hop_rule_expression() -> None:
    expression = "A -[causes]-> B -[causes]-> C => [indirectly_causes]"
    rule = RuleDSL.parse(expression)
    assert rule.relation_sequence == ["causes", "causes"]
    assert rule.inferred_relation == "indirectly_causes"


def test_apply_multi_hop_rule() -> None:
    topo = Topologist()
    topo.add_edge("A", "causes", "B", confidence=0.9)
    topo.add_edge("B", "causes", "C", confidence=0.9)
    rule = MultiHopRule(relation_sequence=["causes", "causes"], inferred_relation="indirectly_causes")
    created = topo.apply_multi_hop_rule(rule)
    assert created == 1
    assert topo.graph.has_edge("A", "C")


def test_multi_hop_rule_respects_min_confidence() -> None:
    topo = Topologist()
    topo.add_edge("A", "r1", "B", confidence=0.2)
    topo.add_edge("B", "r2", "C", confidence=0.2)
    
    # confidence product: 0.2 * 0.2 = 0.04, which is < 0.5 (default min_confidence)
    rule = MultiHopRule(
        relation_sequence=["r1", "r2"],
        inferred_relation="weak_path",
        min_confidence=0.5
    )
    created = topo.apply_multi_hop_rule(rule)
    assert created == 0
    assert not topo.graph.has_edge("A", "C")


def test_sqlite_persistence_roundtrip(tmp_path: Path) -> None:
    topo = Topologist()
    topo.add_edge("A", "rel", "B", confidence=0.6)
    adapter = SQLitePersistenceAdapter(tmp_path / "topology.db")
    adapter.save(topo)
    loaded = adapter.load()
    assert loaded.graph.number_of_nodes() == topo.graph.number_of_nodes()
    assert loaded.graph.number_of_edges() == topo.graph.number_of_edges()
    adapter.close()


def test_sqlite_roundtrip_preserves_config_and_vectors(tmp_path: Path) -> None:
    """SQLite persistence should preserve config and HDC item memory."""
    topo = Topologist(TopologistConfig(dim=1024, seed=123))
    topo.add_edge("A", "rel", "B")
    before_state = topo.update_global_state().copy()
    
    db_path = tmp_path / "topology.db"
    adapter = SQLitePersistenceAdapter(db_path)
    adapter.save(topo)
    loaded = adapter.load()
    adapter.close()
    
    # Check config preserved
    assert loaded.config.dim == 1024
    assert loaded.config.seed == 123
    
    # Check vectors preserved (recovered with same seed)
    after_state = loaded.update_global_state()
    assert loaded.hdc.similarity(before_state, after_state) == 1.0


def test_approximate_nearest_neighbor_returns_known_node() -> None:
    topo = Topologist()
    topo.add_node("Memory", "concept")
    topo.add_node("Model", "concept")
    ann = ApproximateNearestNeighbor(topo)
    ann.build_index()
    nearest = ann.query("Memory", top_k=1)
    assert nearest[0][0] == "Memory"


def test_local_agent_memory_stores_summary() -> None:
    topo = Topologist()
    topo.add_edge("A", "rel", "B")
    adapter = LocalLLMAgentMemoryAdapter()
    adapter.persist(topo)
    assert "A-[rel]->B" in adapter.memory[-1]


def test_claude_openclaw_adapter_without_key() -> None:
    claude = ClaudeCodeAdapter()
    openclaw = OpenClawAdapter()
    assert "API key" in claude.query("test")
    assert "API key" in openclaw.query("test")


def test_rule_dsl_fails_on_invalid_expression() -> None:
    with pytest.raises(ValueError):
        RuleDSL.parse("invalid rule")


def test_version_matches_package_metadata() -> None:
    """Ensure __version__ matches pyproject.toml."""
    import topologist
    assert topologist.__version__ == "0.1.11"
