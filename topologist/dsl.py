from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MultiHopRule(BaseModel):
    """A multi-hop inference rule expressed as a sequence of relations."""

    model_config = ConfigDict(extra="forbid")

    relation_sequence: list[str] = Field(min_length=1)
    inferred_relation: str = Field(min_length=1)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("relation_sequence", mode="before")
    @classmethod
    def normalize_relations(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value


class RuleDSL:
    RELATION_PATTERN = re.compile(r"-\[\s*([^\]]+?)\s*\]->")
    INFERRED_PATTERN = re.compile(r"=>\s*\[\s*([^\]]+?)\s*\]")

    @classmethod
    def parse(cls, expression: str) -> MultiHopRule:
        if not expression or "=>" not in expression:
            raise ValueError("Invalid rule expression. Expected syntax like 'A -[rel1]-> B -[rel2]-> C => [inferred_relation]'.")
        relations = cls.RELATION_PATTERN.findall(expression)
        inferred = cls.INFERRED_PATTERN.search(expression)
        if not relations or inferred is None:
            raise ValueError("Could not parse the rule expression. Ensure relations and inferred relation are present.")
        return MultiHopRule(
            relation_sequence=[relation.strip() for relation in relations],
            inferred_relation=inferred.group(1).strip(),
        )
