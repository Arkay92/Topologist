from __future__ import annotations

from topologist import Topologist
from topologist.service import create_app


def test_service_health_and_topology_endpoints() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(create_app())

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/topology").json() == {"nodes": 0, "edges": 0, "communities": []}


def test_service_ingest_and_nearest_endpoints() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(create_app())

    response = client.post(
        "/ingest",
        json={
            "source": "HDC",
            "relation": "models",
            "target": "Memory",
            "confidence": 0.8,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"success": True}
    assert client.get("/topology").json()["edges"] == 1
    nearest = client.get("/nearest/Memory", params={"top_k": 1}).json()["nearest"]
    assert nearest[0][0] == "Memory"


def test_service_rule_endpoint_applies_dsl_and_reports_parse_errors() -> None:
    from fastapi.testclient import TestClient

    topo = Topologist()
    topo.add_edge("A", "causes", "B", confidence=0.9)
    topo.add_edge("B", "causes", "C", confidence=0.9)
    client = TestClient(create_app(topo))

    response = client.post(
        "/rule",
        json={"rule_expression": "A -[causes]-> B -[causes]-> C => [indirectly_causes]"},
    )
    invalid = client.post("/rule", json={"rule_expression": "invalid"})

    assert response.status_code == 200
    assert response.json() == {"created": 1}
    assert invalid.status_code == 400
