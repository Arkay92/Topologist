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
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS hdc_memory (
                    symbol TEXT PRIMARY KEY,
                    vector TEXT NOT NULL
                )
                """
            )
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
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vector TEXT NOT NULL
                )
                """
            )

    def save(self, topology: "Topologist") -> None:
        from topologist.engine import Topologist

        with self.connection:
            self.connection.execute("DELETE FROM config")
            self.connection.execute("DELETE FROM hdc_memory")
            self.connection.execute("DELETE FROM nodes")
            self.connection.execute("DELETE FROM edges")
            self.connection.execute("DELETE FROM snapshots")

            # save config
            for key, value in topology.config.model_dump().items():
                self.connection.execute(
                    "INSERT INTO config (key, value) VALUES (?, ?)",
                    (key, json.dumps(value)),
                )

            # save HDC item memory
            for symbol, vector in topology.hdc.item_memory.items():
                self.connection.execute(
                    "INSERT INTO hdc_memory (symbol, vector) VALUES (?, ?)",
                    (symbol, json.dumps(vector.tolist())),
                )

            # save nodes
            for node, data in topology.graph.nodes(data=True):
                self.connection.execute(
                    "INSERT INTO nodes (name, kind, metadata) VALUES (?, ?, ?)",
                    (node, data.get("kind", "concept"), json.dumps(data.get("metadata", {}))),
                )

            # save edges
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

            # save snapshots
            for snapshot in topology.snapshots:
                self.connection.execute(
                    "INSERT INTO snapshots (vector) VALUES (?)",
                    (json.dumps(snapshot.tolist()),),
                )

    def load(self) -> "Topologist":
        from topologist.engine import Topologist
        from topologist.hdc import HyperVectorSpace

        cursor = self.connection.cursor()

        # load config
        config_dict = {}
        for key, value_json in cursor.execute("SELECT key, value FROM config"):
            config_dict[key] = json.loads(value_json)

        # create topology with loaded config
        if config_dict:
            from topologist.config import TopologistConfig
            config = TopologistConfig(**config_dict)
        else:
            from topologist.config import TopologistConfig
            config = TopologistConfig()

        topology = Topologist(config=config)

        # load HDC item memory
        for symbol, vector_json in cursor.execute("SELECT symbol, vector FROM hdc_memory"):
            import numpy as np
            topology.hdc.item_memory[symbol] = np.asarray(json.loads(vector_json), dtype=np.int8)

        # load nodes
        for name, kind, metadata_json in cursor.execute("SELECT name, kind, metadata FROM nodes"):
            topology.add_node(name, kind, **json.loads(metadata_json))

        # load edges
        for source, relation, target, weight, confidence, metadata_json in cursor.execute(
            "SELECT source, relation, target, weight, confidence, metadata FROM edges"
        ):
            topology.add_edge(
                source,
                relation,
                target,
                weight=weight,
                confidence=confidence,
                **json.loads(metadata_json),
            )

        # load snapshots
        import numpy as np
        snapshots = []
        for (vector_json,) in cursor.execute("SELECT vector FROM snapshots ORDER BY id"):
            snapshots.append(np.asarray(json.loads(vector_json), dtype=np.int8))
        topology.snapshots = snapshots

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
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS hdc_memory (
                    symbol TEXT PRIMARY KEY,
                    vector TEXT NOT NULL
                )
                """
            )
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
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    id SERIAL PRIMARY KEY,
                    vector TEXT NOT NULL
                )
                """
            )
            self.connection.commit()

    def save(self, topology: "Topologist") -> None:
        with self.connection.cursor() as cursor:
            cursor.execute("DELETE FROM config")
            cursor.execute("DELETE FROM hdc_memory")
            cursor.execute("DELETE FROM nodes")
            cursor.execute("DELETE FROM edges")
            cursor.execute("DELETE FROM snapshots")

            # save config
            for key, value in topology.config.model_dump().items():
                cursor.execute(
                    "INSERT INTO config (key, value) VALUES (%s, %s)",
                    (key, json.dumps(value)),
                )

            # save HDC item memory
            for symbol, vector in topology.hdc.item_memory.items():
                cursor.execute(
                    "INSERT INTO hdc_memory (symbol, vector) VALUES (%s, %s)",
                    (symbol, json.dumps(vector.tolist())),
                )

            # save nodes
            for node, data in topology.graph.nodes(data=True):
                cursor.execute(
                    "INSERT INTO nodes (name, kind, metadata) VALUES (%s, %s, %s)",
                    (node, data.get("kind", "concept"), json.dumps(data.get("metadata", {}))),
                )

            # save edges
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

            # save snapshots
            for snapshot in topology.snapshots:
                cursor.execute(
                    "INSERT INTO snapshots (vector) VALUES (%s)",
                    (json.dumps(snapshot.tolist()),),
                )

            self.connection.commit()

    def load(self) -> "Topologist":
        with self.connection.cursor() as cursor:
            # load config
            config_dict = {}
            cursor.execute("SELECT key, value FROM config")
            for key, value_json in cursor.fetchall():
                config_dict[key] = json.loads(value_json)

            # create topology with loaded config
            if config_dict:
                from topologist.config import TopologistConfig
                config = TopologistConfig(**config_dict)
            else:
                from topologist.config import TopologistConfig
                config = TopologistConfig()

            topology = self._import_topologist(config)

            # load HDC item memory
            cursor.execute("SELECT symbol, vector FROM hdc_memory")
            import numpy as np
            for symbol, vector_json in cursor.fetchall():
                topology.hdc.item_memory[symbol] = np.asarray(json.loads(vector_json), dtype=np.int8)

            # load nodes
            cursor.execute("SELECT name, kind, metadata FROM nodes")
            for name, kind, metadata_json in cursor.fetchall():
                topology.add_node(name, kind, **json.loads(metadata_json))

            # load edges
            cursor.execute(
                "SELECT source, relation, target, weight, confidence, metadata FROM edges"
            )
            for source, relation, target, weight, confidence, metadata_json in cursor.fetchall():
                topology.add_edge(
                    source,
                    relation,
                    target,
                    weight=weight,
                    confidence=confidence,
                    **json.loads(metadata_json),
                )

            # load snapshots
            cursor.execute("SELECT vector FROM snapshots ORDER BY id")
            snapshots = []
            for (vector_json,) in cursor.fetchall():
                snapshots.append(np.asarray(json.loads(vector_json), dtype=np.int8))
            topology.snapshots = snapshots

            topology.update_global_state()
            return topology

    def close(self) -> None:
        self.connection.close()

    def _import_topologist(self, config: "TopologistConfig") -> "Topologist":
        from topologist.engine import Topologist

        return Topologist(config=config)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
