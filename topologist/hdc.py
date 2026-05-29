from __future__ import annotations

import json
from pathlib import Path

import math
import numpy as np
from numpy.typing import NDArray

BipolarVector = NDArray[np.int8]


class HyperVectorSpace:
    """Bipolar hyperdimensional vector space.

    Operations:
    - item memory: stable random vectors per symbol
    - bind: elementwise multiplication
    - bundle: majority superposition
    - permute: cyclic shift for order/role encoding
    """

    def __init__(self, dim: int = 10_000, seed: int = 42) -> None:
        if dim < 128:
            raise ValueError("dim must be at least 128")
        self.dim = dim
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.item_memory: dict[str, BipolarVector] = {}

    def random(self) -> BipolarVector:
        return self._rng.choice([-1, 1], size=self.dim).astype(np.int8)

    def get(self, symbol: str) -> BipolarVector:
        key = symbol.strip()
        if not key:
            raise ValueError("symbol cannot be blank")
        if key not in self.item_memory:
            self.item_memory[key] = self.random()
        return self.item_memory[key]

    @staticmethod
    def bind(a: BipolarVector, b: BipolarVector) -> BipolarVector:
        return (a * b).astype(np.int8)

    def bundle(self, vectors: list[BipolarVector]) -> BipolarVector:
        if not vectors:
            return np.zeros(self.dim, dtype=np.int8)
        stacked = np.vstack(vectors).astype(np.int16)
        summed = np.sum(stacked, axis=0)
        return np.where(summed >= 0, 1, -1).astype(np.int8)

    @staticmethod
    def permute(vector: BipolarVector, shifts: int = 1) -> BipolarVector:
        return np.roll(vector, shifts).astype(np.int8)

    @staticmethod
    def similarity(a: BipolarVector, b: BipolarVector) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a.astype(np.float32), b.astype(np.float32)) / denom)

    def encode_relation(self, source: str, relation: str, target: str) -> BipolarVector:
        return self.bind(
            self.bind(self.get(f"node::{source}"), self.get(f"relation::{relation}")),
            self.get(f"node::{target}"),
        )

    def encode_typed_node(self, name: str, kind: str) -> BipolarVector:
        return self.bind(self.get(f"kind::{kind}"), self.get(f"node::{name}"))

    def encode_sequence(self, symbols: list[str]) -> BipolarVector:
        vectors = []
        position = self.get("role::position")
        for idx, symbol in enumerate(symbols):
            role = self.permute(position, idx)
            vectors.append(self.bind(role, self.get(symbol)))
        return self.bundle(vectors)

    def encode_confidence(self, confidence: float) -> BipolarVector:
        """Encode a continuous confidence value as a permuted base vector.

        Confidence is expected in [0.0, 1.0]. We create a base confidence
        vector in item memory and apply a cyclic permutation proportional to
        the confidence value. This gives a smooth, continuous encoding that
        changes as confidence decays.
        """
        try:
            c = float(confidence)
        except Exception:
            c = 1.0
        c = max(0.0, min(1.0, c))
        base = self.get("confidence::base")
        # map confidence to integer shift within [0, dim-1]
        # avoid round() overload ambiguity by using explicit float math
        shift = math.floor(float(c) * (self.dim - 1) + 0.5)
        return self.permute(base, shift)

    def to_jsonable(self) -> dict[str, object]:
        return {
            "dim": self.dim,
            "seed": self.seed,
            "item_memory": {key: value.tolist() for key, value in self.item_memory.items()},
        }

    @classmethod
    def from_jsonable(cls, payload: dict[str, object]) -> "HyperVectorSpace":
        space = cls(dim=int(str(payload["dim"])), seed=int(str(payload["seed"])))
        raw_memory = payload.get("item_memory", {})
        if not isinstance(raw_memory, dict):
            raise ValueError("item_memory must be an object")
        space.item_memory = {
            str(key): np.asarray(value, dtype=np.int8) for key, value in raw_memory.items()
        }
        return space

    def save_item_memory(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_jsonable()), encoding="utf-8")

    @classmethod
    def load_item_memory(cls, path: str | Path) -> "HyperVectorSpace":
        return cls.from_jsonable(json.loads(Path(path).read_text(encoding="utf-8")))
