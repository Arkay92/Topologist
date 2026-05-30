from __future__ import annotations

from typing import Any

from topologist.engine import Topologist


def create_app(topology: Topologist | None = None) -> Any:
    try:
        from fastapi import FastAPI, Body, HTTPException  # type: ignore
        from pydantic import BaseModel  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "FastAPI is required to create the Topologist service. "
            "Install with `pip install topologist[api]`."
        ) from exc

    topology = topology or Topologist()
    app = FastAPI(title="Topologist Service")

    from pydantic import Field

    class IngestEvent(BaseModel):
        source: str
        relation: str
        target: str
        weight: float | None = None
        confidence: float | None = None
        metadata: dict[str, Any] = Field(default_factory=dict)

    class RuleRequest(BaseModel):
        rule_expression: str

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ingest")
    def ingest(event: IngestEvent) -> dict[str, Any]:
        result = topology.ingest_event(event.model_dump())
        return {"success": result}

    @app.post("/rule")
    def apply_rule(rule_request: RuleRequest) -> dict[str, Any]:
        from topologist.dsl import RuleDSL

        try:
            count = topology.apply_dsl_rule(rule_request.rule_expression)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"created": count}

    @app.get("/nearest/{query}")
    def nearest(query: str, top_k: int = 5) -> dict[str, Any]:
        return {"nearest": topology.nearest_nodes(query, top_k=top_k)}

    @app.get("/topology")
    def topology_summary() -> dict[str, Any]:
        return {
            "nodes": topology.graph.number_of_nodes(),
            "edges": topology.graph.number_of_edges(),
            "communities": topology.communities(),
        }

    return app
