from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from topologist.models import EdgeRecord, NodeRecord


class PersistenceAdapter(ABC):
    @abstractmethod
    def save(self, topology: "Topologist") -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> "Topologist":
        raise NotImplementedError


class SQLitePersistenceAdapter(PersistenceAdapter):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.connection = sqlite3.connect(self.path)
        self._initialize()

    def _initialize(self) -> None:
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    name TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS edges (
                    source TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    target TEXT NOT NULL,
                    weight REAL NOT NULL,
                    confidence REAL NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )

    def save(self, topology: "Topologist") -> None:
        from topologist.engine import Topologist

        with self.connection:
            self.connection.execute("DELETE FROM nodes")
            self.connection.execute("DELETE FROM edges")
            for node, data in topology.graph.nodes(data=True):
                self.connection.execute(
                    "INSERT INTO nodes (name, kind, metadata) VALUES (?, ?, ?)",
                    (node, data.get("kind", "concept"), json.dumps(data.get("metadata", {}))),
                )
            for source, target, data in topology.graph.edges(data=True):
                self.connection.execute(
                    "INSERT INTO edges (source, relation, target, weight, confidence, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        source,
                        data.get("relation", "related_to"),
                        target,
                        float(data.get("weight", 1.0)),
                        float(data.get("confidence", 1.0)),
                        json.dumps(data.get("metadata", {})),
                    ),
                )

    def load(self) -> "Topologist":
        from topologist.engine import Topologist

        topology = Topologist()
        cursor = self.connection.cursor()
        for name, kind, metadata_json in cursor.execute("SELECT name, kind, metadata FROM nodes"):
            metadata = json.loads(metadata_json)
            topology.add_node(name, kind, **metadata)
        for source, relation, target, weight, confidence, metadata_json in cursor.execute(
            "SELECT source, relation, target, weight, confidence, metadata FROM edges"
        ):
            metadata = json.loads(metadata_json)
            topology.add_edge(
                source,
                relation,
                target,
                weight=weight,
                confidence=confidence,
                **metadata,
            )
        topology.update_global_state()
        return topology

    def close(self) -> None:
        self.connection.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class PostgresPersistenceAdapter(PersistenceAdapter):
    def __init__(self, dsn: str) -> None:
        try:
            import psycopg2  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "psycopg2-binary is required for Postgres persistence. "
                "Install with `pip install topologist[db]`."
            ) from exc
        self._psycopg2 = psycopg2
        self.dsn = dsn
        self.connection = self._psycopg2.connect(dsn)
        self._initialize()

    def _initialize(self) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    name TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS edges (
                    source TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    target TEXT NOT NULL,
                    weight REAL NOT NULL,
                    confidence REAL NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            self.connection.commit()

    def save(self, topology: "Topologist") -> None:
        with self.connection.cursor() as cursor:
            cursor.execute("DELETE FROM nodes")
            cursor.execute("DELETE FROM edges")
            for node, data in topology.graph.nodes(data=True):
                cursor.execute(
                    "INSERT INTO nodes (name, kind, metadata) VALUES (%s, %s, %s)",
                    (node, data.get("kind", "concept"), json.dumps(data.get("metadata", {}))),
                )
            for source, target, data in topology.graph.edges(data=True):
                cursor.execute(
                    "INSERT INTO edges (source, relation, target, weight, confidence, metadata) VALUES (%s, %s, %s, %s, %s, %s)",
                    (
                        source,
                        data.get("relation", "related_to"),
                        target,
                        float(data.get("weight", 1.0)),
                        float(data.get("confidence", 1.0)),
                        json.dumps(data.get("metadata", {})),
                    ),
                )
            self.connection.commit()

    def load(self) -> "Topologist":
        topology = self._import_topologist()
        cursor = self.connection.cursor()
        cursor.execute("SELECT name, kind, metadata FROM nodes")
        for name, kind, metadata_json in cursor.fetchall():
            topology.add_node(name, kind, **json.loads(metadata_json))
        cursor.execute("SELECT source, relation, target, weight, confidence, metadata FROM edges")
        for source, relation, target, weight, confidence, metadata_json in cursor.fetchall():
            topology.add_edge(
                source,
                relation,
                target,
                weight=weight,
                confidence=confidence,
                **json.loads(metadata_json),
            )
        topology.update_global_state()
        return topology

    def close(self) -> None:
        self.connection.close()

    def _import_topologist(self) -> "Topologist":
        from topologist.engine import Topologist

        return Topologist()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
