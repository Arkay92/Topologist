from __future__ import annotations

import pytest

from topologist import Topologist
from topologist.dsl import MultiHopRule, RuleDSL


def test_rule_dsl_parses_relation_sequence_and_inferred_relation() -> None:
    rule = RuleDSL.parse("A -[ causes ]-> B -[supports]-> C => [ indirectly_supports ]")

    assert rule.relation_sequence == ["causes", "supports"]
    assert rule.inferred_relation == "indirectly_supports"
    assert rule.min_confidence == 0.5


def test_rule_dsl_rejects_invalid_expressions() -> None:
    with pytest.raises(ValueError, match="Expected syntax"):
        RuleDSL.parse("A causes B")

    with pytest.raises(ValueError, match="relations and inferred relation"):
        RuleDSL.parse("A -[causes]-> B =>")


def test_multi_hop_rule_normalizes_comma_separated_relations() -> None:
    rule = MultiHopRule(
        relation_sequence="causes, supports, enables",
        inferred_relation="indirectly_enables",
        min_confidence=0.7,
    )

    assert rule.relation_sequence == ["causes", "supports", "enables"]
    assert rule.min_confidence == 0.7


def test_apply_dsl_rule_uses_default_min_confidence() -> None:
    topo = Topologist()
    topo.add_edge("A", "causes", "B", confidence=0.9)
    topo.add_edge("B", "causes", "C", confidence=0.9)

    created = topo.apply_dsl_rule("A -[causes]-> B -[causes]-> C => [indirectly_causes]")

    assert created == 1
    inferred = [
        (source, target, data)
        for source, target, data in topo.graph.edges(data=True)
        if data["relation"] == "indirectly_causes"
    ]
    assert len(inferred) == 1
    assert inferred[0][0] == "A"
    assert inferred[0][1] == "C"
    assert inferred[0][2]["confidence"] == pytest.approx(0.81)


def test_multi_hop_rule_respects_min_confidence() -> None:
    topo = Topologist()
    topo.add_edge("A", "causes", "B", confidence=0.4)
    topo.add_edge("B", "causes", "C", confidence=0.4)
    rule = MultiHopRule(
        relation_sequence=["causes", "causes"],
        inferred_relation="indirectly_causes",
        min_confidence=0.5,
    )

    assert topo.apply_multi_hop_rule(rule) == 0
    assert not topo.graph.has_edge("A", "C")
