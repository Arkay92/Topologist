from __future__ import annotations

from pathlib import Path

from topologist.engine import Topologist


def export_mermaid(topology: Topologist, path: str | Path) -> None:
    """Export graph as a Mermaid flowchart."""
    lines = ["flowchart LR"]
    for source, target, data in topology.graph.edges(data=True):
        relation = str(data.get("relation", "related_to")).replace('"', "'")
        lines.append(f'    "{source}" -- "{relation}" --> "{target}"')
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
