from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from topologist.engine import Topologist
from topologist.models import ReasoningRule
from topologist.visualization import export_mermaid

app = typer.Typer(help="Topologist CLI")
console = Console()


@app.command()
def demo(output: Path = Path("topology.json")) -> None:
    """Create a demo hyperdimensional neuro-symbolic topology."""
    topo = Topologist()
    topo.add_node("Neuron", "biological_system")
    topo.add_node("Synapse", "biological_system")
    topo.add_node("Memory", "cognitive_system")
    topo.add_node("HDC", "computational_model")
    topo.add_node("KnowledgeGraph", "symbolic_model")
    topo.add_node("Reasoning", "cognitive_process")
    topo.add_edge("Neuron", "connects_to", "Synapse", confidence=0.95)
    topo.add_edge("Synapse", "supports", "Memory", confidence=0.90)
    topo.add_edge("HDC", "models", "Memory", confidence=0.85)
    topo.add_edge("KnowledgeGraph", "supports", "Reasoning", confidence=0.92)
    topo.add_edge("HDC", "enhances", "KnowledgeGraph", confidence=0.80)
    topo.add_edge("Reasoning", "uses", "Memory", confidence=0.88)
    topo.apply_rule(
        ReasoningRule(
            relation_a="connects_to",
            relation_b="supports",
            inferred_relation="indirectly_supports",
            min_confidence=0.5,
        )
    )
    topo.update_global_state(take_snapshot=True)
    topo.save(output)
    console.print(f"[green]Saved demo topology to {output}[/green]")


@app.command()
def inspect(path: Path) -> None:
    """Print summary metrics for a saved topology."""
    topo = Topologist.load(path)
    table = Table(title="Topologist Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Nodes", str(topo.graph.number_of_nodes()))
    table.add_row("Edges", str(topo.graph.number_of_edges()))
    table.add_row("Communities", json.dumps(topo.communities()))
    table.add_row("Drift", f"{topo.topology_drift():.4f}")
    console.print(table)


@app.command()
def mermaid(path: Path, output: Path = Path("topology.mmd")) -> None:
    """Export a saved topology as Mermaid."""
    topo = Topologist.load(path)
    export_mermaid(topo, output)
    console.print(f"[green]Saved Mermaid graph to {output}[/green]")


if __name__ == "__main__":
    app()
