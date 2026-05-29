from __future__ import annotations

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
