from topologist import Topologist
from topologist.models import ReasoningRule
from topologist.visualization import export_mermaid


def main() -> None:
    topo = Topologist()

    topo.add_node("Agent", "system")
    topo.add_node("Memory", "cognitive_layer")
    topo.add_node("KnowledgeGraph", "symbolic_layer")
    topo.add_node("HypervectorSpace", "hdc_layer")
    topo.add_node("AnomalyDetector", "reasoning_layer")

    topo.add_edge("Agent", "uses", "Memory", confidence=0.95)
    topo.add_edge("Memory", "encoded_by", "HypervectorSpace", confidence=0.90)
    topo.add_edge("Memory", "structured_by", "KnowledgeGraph", confidence=0.88)
    topo.add_edge("KnowledgeGraph", "feeds", "AnomalyDetector", confidence=0.82)

    inferred = topo.apply_rule(
        ReasoningRule(
            relation_a="uses",
            relation_b="encoded_by",
            inferred_relation="indirectly_uses",
            min_confidence=0.5,
        )
    )

    topo.update_global_state(take_snapshot=True)

    print(f"Inferred edges: {inferred}")
    print("Centrality:", topo.centrality())
    print("Communities:", topo.communities())
    print("Nearest to Memory:", topo.nearest_nodes("Memory", top_k=3))
    print(
        "Anomaly score:",
        topo.relation_anomaly_score("Agent", "ignores", "Memory"),
    )

    topo.save("example_topology.json")
    export_mermaid(topo, "example_topology.mmd")


if __name__ == "__main__":
    main()
