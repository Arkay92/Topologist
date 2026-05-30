from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from topologist.cli import app


def test_cli_demo_writes_topology(tmp_path: Path) -> None:
    runner = CliRunner()
    output = tmp_path / "topology.json"

    result = runner.invoke(app, ["demo", "--output", str(output)])

    assert result.exit_code == 0
    assert output.exists()
    assert "Saved demo topology" in result.output


def test_cli_inspect_reads_saved_topology(tmp_path: Path) -> None:
    runner = CliRunner()
    output = tmp_path / "topology.json"
    assert runner.invoke(app, ["demo", "--output", str(output)]).exit_code == 0

    result = runner.invoke(app, ["inspect", str(output)])

    assert result.exit_code == 0
    assert "Topologist Summary" in result.output
    assert "Nodes" in result.output
    assert "Edges" in result.output


def test_cli_mermaid_exports_graph(tmp_path: Path) -> None:
    runner = CliRunner()
    topology_path = tmp_path / "topology.json"
    mermaid_path = tmp_path / "topology.mmd"
    assert runner.invoke(app, ["demo", "--output", str(topology_path)]).exit_code == 0

    result = runner.invoke(
        app,
        ["mermaid", str(topology_path), "--output", str(mermaid_path)],
    )

    assert result.exit_code == 0
    assert mermaid_path.exists()
    assert "Saved Mermaid graph" in result.output
