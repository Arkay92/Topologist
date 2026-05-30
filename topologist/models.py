from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NodeRecord(BaseModel):
    """Serializable symbolic node record."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: str = Field(default="concept", min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "kind")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value cannot be blank")
        return value


class EdgeRecord(BaseModel):
    """Serializable symbolic edge record."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    target: str = Field(min_length=1)
    weight: float = Field(default=1.0, ge=0.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: "ProvenanceRecord" = Field(default_factory=lambda: ProvenanceRecord())

    @field_validator("source", "relation", "target")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value cannot be blank")
        return value


class ReasoningRule(BaseModel):
    """A simple two-hop symbolic inference rule.

    Example:
        if A --connects_to--> B and B --supports--> C,
        infer A --indirectly_supports--> C.
    """

    model_config = ConfigDict(extra="forbid")

    relation_a: str = Field(min_length=1)
    relation_b: str = Field(min_length=1)
    inferred_relation: str = Field(min_length=1)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ProvenanceRecord(BaseModel):
    """Provenance and trust metadata for an edge."""

    model_config = ConfigDict(extra="forbid")

    source_type: str = Field(default="api", min_length=1)
    evidence: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    reinforcement_count: int = Field(default=1, ge=1)
    decay_policy: str = Field(default="default", min_length=1)
    trust_score: float = Field(default=1.0, ge=0.0, le=1.0)
    derived: bool = False
    rule: dict[str, Any] | None = None
    evidence_path: list[list[str]] = Field(default_factory=list)
