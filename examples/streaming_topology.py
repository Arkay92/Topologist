from time import sleep
from random import random

from topologist import Topologist
from topologist.models import ReasoningRule


def streaming_demo(events=5, pause=0.2):
    topo = Topologist()

    # seeded example nodes
    topo.add_node("Sensor", "device")
    topo.add_node("Alert", "event")
    topo.add_node("Host", "asset")

    rule = ReasoningRule(
        relation_a="reports",
        relation_b="triggers",
        inferred_relation="may_alert",
        min_confidence=0.4,
    )

    for i in range(events):
        # simulate an incoming observation
        src = "Sensor"
        rel = "reports"
        tgt = "Host"
        confidence = 0.6 + 0.4 * random()
        topo.add_edge(src, rel, tgt, confidence=confidence)

        # sometimes add a trigger relation
        if random() > 0.7:
            topo.add_edge(tgt, "triggers", "Alert", confidence=0.85)

        # apply reasoning and snapshot
        created = topo.apply_rule(rule)
        topo.update_global_state(take_snapshot=True)

        # compute drift and anomalies
        drift = topo.topology_drift()
        anomalous = topo.is_anomalous_relation(src, "may_alert", "Alert")

        print(f"event={i} confidence={confidence:.2f} inferred={created} drift={drift:.4f} anomalous={anomalous}")
        sleep(pause)

    print("Final centrality:", topo.centrality())


if __name__ == "__main__":
    streaming_demo(events=10, pause=0.05)
