from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict


class TopologistConfig(BaseModel):
    """Runtime configuration for the topology engine."""

    model_config = ConfigDict(frozen=True)

    dim: int = Field(default=10_000, ge=128, description="Hypervector dimensionality.")
    seed: int = Field(default=42, description="Random seed for reproducible item memory.")
    similarity_floor: float = Field(default=-1.0, ge=-1.0, le=1.0)
    anomaly_threshold: float = Field(default=0.85, ge=0.0, le=2.0)
    decay_rate: float = Field(default=0.98, gt=0.0, le=1.0)
    max_snapshots: int = Field(default=100, ge=1)
    default_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    default_weight: float = Field(default=1.0, ge=0.0)
