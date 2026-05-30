from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from topologist import Topologist, TopologistConfig
from topologist.persistence import PostgresPersistenceAdapter, SQLitePersistenceAdapter


def test_sqlite_roundtrip_preserves_config_hdc_memory_and_snapshots(tmp_path: Path) -> None:
    topo = Topologist(TopologistConfig(dim=1024, seed=123, max_snapshots=3))
    topo.add_node("Memory", "concept", domain="cognitive")
    topo.add_edge("HDC", "models", "Memory", confidence=0.8, evidence_source="paper")
    expected_state = topo.update_global_state(take_snapshot=True).copy()
    expected_memory = {key: value.copy() for key, value in topo.hdc.item_memory.items()}

    adapter = SQLitePersistenceAdapter(tmp_path / "topology.db")
    adapter.save(topo)
    loaded = adapter.load()
    adapter.close()

    assert loaded.config == topo.config
    assert loaded.graph.number_of_nodes() == topo.graph.number_of_nodes()
    assert loaded.graph.number_of_edges() == topo.graph.number_of_edges()
    assert len(loaded.snapshots) == 1
    assert loaded.hdc.item_memory.keys() == expected_memory.keys()
    for key, vector in expected_memory.items():
        assert (loaded.hdc.item_memory[key] == vector).all()
    assert loaded.hdc.similarity(expected_state, loaded.update_global_state()) == pytest.approx(1.0)


def test_sqlite_save_replaces_previous_rows(tmp_path: Path) -> None:
    path = tmp_path / "topology.db"
    adapter = SQLitePersistenceAdapter(path)

    first = Topologist()
    first.add_edge("A", "rel", "B")
    adapter.save(first)

    second = Topologist()
    second.add_edge("X", "rel", "Y")
    adapter.save(second)
    adapter.close()

    with sqlite3.connect(path) as connection:
        nodes = {row[0] for row in connection.execute("SELECT name FROM nodes")}
        edges = list(connection.execute("SELECT source, relation, target FROM edges"))

    assert nodes == {"X", "Y"}
    assert edges == [("X", "rel", "Y")]


def test_topologist_sqlite_helpers_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "topology.db"
    topo = Topologist(TopologistConfig(dim=1024, seed=99))
    topo.add_edge("A", "supports", "B", confidence=0.7)
    topo.update_global_state(take_snapshot=True)

    topo.save_to_sqlite(path)
    loaded = Topologist.load_from_sqlite(path)

    assert loaded.config == topo.config
    assert loaded.neighbors("A")[0]["confidence"] == pytest.approx(0.7)
    assert len(loaded.snapshots) == 1


def test_postgres_adapter_reports_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    original_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "psycopg2":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"topologist\[db\]"):
        PostgresPersistenceAdapter("postgresql://example")
