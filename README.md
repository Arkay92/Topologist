# Topologist

<p align="center">
  <img width="300" height="300" alt="Topologist logo" src="https://github.com/Arkay92/Topologist/raw/main/assets/Topologist.png" />
</p>

<p align="center">
  <a href="https://github.com/Arkay92/Topologist/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Arkay92/Topologist/actions/workflows/ci.yml/badge.svg" /></a>
  <img alt="Coverage" src="https://img.shields.io/badge/coverage-74%25-yellow" />
  <a href="https://pypi.org/project/topologist/"><img alt="PyPI" src="https://img.shields.io/pypi/v/topologist.svg" /></a>
  <a href="https://pepy.tech/project/topologist"><img alt="Downloads" src="https://static.pepy.tech/badge/topologist" /></a>
  <img alt="Python" src="https://img.shields.io/pypi/pyversions/topologist.svg" />
  <img alt="License" src="https://img.shields.io/pypi/l/topologist.svg" />
</p>

A production-hardened **hyperdimensional neuro-symbolic topology system** in Python.

Topologist combines:

- **Hyperdimensional Computing / Vector Symbolic Architecture** for robust distributed representations.
- **Neuro-symbolic graph topology** using NetworkX.
- **Rule-based inference** over symbolic relations.
- **Topology analytics** including PageRank centrality, communities, shortest paths, drift, and anomaly scoring.
- **Provenance-first memory** with edge evidence, trust, reinforcement count, timestamps, and explanations.
- **Persistence and export** to JSON, SQLite, Postgres, GraphML, and Mermaid.
- **CLI, FastAPI, streaming, ANN, tracing, and agent-memory adapters** for production-facing workflows.

---

## Why this exists

Most symbolic graph systems are explainable but brittle. Most neural/vector systems are robust but opaque.

Topologist sits between the two:

```text
Symbolic entities and relations
  -> hyperdimensional encoding
  -> graph topology
  -> reasoning + analytics + anomaly detection
  -> persistence, streaming, service, and visualization layers
```

Each node and relation is stored symbolically, but also encoded into a high-dimensional bipolar hypervector. This gives you a graph that is queryable and explainable while also having a distributed topology-level memory state.

## Architecture

```text
events / CSV / API / CLI
        |
        v
Topologist engine
  - NetworkX MultiDiGraph
  - HDC item memory
  - confidence-aware global state
        |
        +--> DSL and multi-hop inference
        +--> anomaly scoring and drift detection
        +--> ANN nearest-neighbor lookup
        +--> JSON / SQLite / Postgres persistence
        +--> Mermaid / GraphML export
        +--> FastAPI / Kafka / Redis / WebSocket adapters
```

The HDC layer uses stable namespaced item memory such as `node::Host`, `relation::connects_to`, and quantized `confidence::band::N` vectors. Candidate relations are scored by unbinding the relation signal and comparing it with both relation prototypes and local topology.

---

## Install

```bash
pip install topologist
```

For development:

```bash
pip install -e ".[dev]"
python examples/demo.py
```

---

## Quick Start

```python
from topologist import Topologist
from topologist.models import ReasoningRule

system = Topologist()

system.add_edge("Neuron", "connects_to", "Synapse", confidence=0.95)
system.add_edge("Synapse", "supports", "Memory", confidence=0.90)
system.add_edge("HDC", "models", "Memory", confidence=0.85)

created = system.apply_rule(
    ReasoningRule(
        relation_a="connects_to",
        relation_b="supports",
        inferred_relation="indirectly_supports",
        min_confidence=0.5,
    )
)

system.update_global_state(take_snapshot=True)

print("Created inferred edges:", created)
print("Centrality:", system.centrality())
print("Communities:", system.communities())
print("Nearest nodes:", system.nearest_nodes("Memory"))
print("Path:", system.shortest_path("Neuron", "Memory"))

system.save("topology.json")
```

---

## Worked Example: Cyber Event Topology

Run a deterministic scenario that ingests security events, infers multi-hop risk, scores anomalies, measures drift, and exports Mermaid:

```bash
python examples/cyber_event_topology.py
```

The example models a path like:

```text
IP -> connects_to -> service
service -> exposes -> vulnerability
vulnerability -> enables -> privilege_escalation
```

Then it infers:

```text
IP -> may_escalate_via -> privilege_escalation
```

The script writes:

- `cyber_event_topology.json`
- `cyber_event_topology.mmd`

There is also a smaller streaming simulation:

```bash
python examples/streaming_topology.py
```

---

## CLI

Create a demo topology:

```bash
topologist demo --output topology.json
```

Inspect it:

```bash
topologist inspect topology.json
```

Export Mermaid:

```bash
topologist mermaid topology.json --output topology.mmd
```

---

## Main Features

### 1. Hyperdimensional item memory

Stable symbols are encoded into bipolar vectors:

```text
symbol -> {-1, +1}^D
```

The engine supports binding, bundling, permutation, quantized confidence bands, and cosine similarity.

### 2. Symbolic topology graph

The graph is a `networkx.MultiDiGraph`, so it supports multiple relation types between the same source and target.

```text
HDC --models--> Memory
HDC --enhances--> KnowledgeGraph
KnowledgeGraph --supports--> Reasoning
```

### 3. Rule-based inference

Rules operate over two-hop motifs and general multi-hop DSL expressions:

```text
A --relation_a--> B
B --relation_b--> C
----------------------
A --inferred_relation--> C
```

```python
topology.apply_dsl_rule("A -[causes]-> B -[causes]-> C => [indirectly_causes]")
```

### 4. Drift detection

The global graph state is bundled into a single hypervector snapshot.

```python
system.update_global_state(take_snapshot=True)
drift = system.topology_drift()
```

### 5. Anomaly scoring

Candidate relations can be compared against local topology and relation prototypes using unbinding:

```python
score = system.relation_anomaly_score("A", "unexpected_relation", "B")
```

The scoring method:

1. Unbinds the candidate relation to recover the relation signal.
2. Compares against the expected relation hypervector.
3. Checks similarity to local topology edges.
4. Returns anomaly in `[0, 1]`, where `1` is most anomalous.

### 6. Persistence and service adapters

Topologist supports JSON, SQLite, Postgres, FastAPI, Kafka, Redis Streams, WebSocket ingestion, OpenTelemetry tracing, approximate nearest neighbors, and agent-memory persistence.

### 7. Provenance and reasoning traces

Edges carry provenance metadata so agent memory and streaming systems can explain why a relation exists:

```python
topology.add_edge(
    "A",
    "supports",
    "B",
    source_type="sensor",
    evidence=["event-123"],
    trust_score=0.8,
)

topology.explain_edge("A", "supports", "B")
```

Inferred edges include the rule and evidence path used to derive them.

---

## Project Structure

```text
topologist/
|-- __init__.py
|-- agent.py          # Agent memory adapters
|-- ann.py            # Approximate nearest-neighbor search
|-- bridges.py        # PyTorch Geometric bridge
|-- cli.py            # CLI commands
|-- config.py         # Runtime configuration
|-- dsl.py            # Rule DSL for multi-hop inference
|-- engine.py         # Core topology engine
|-- exceptions.py     # Custom exception types
|-- hdc.py            # Hyperdimensional computing operations
|-- io.py             # CSV utilities
|-- models.py         # Pydantic records
|-- persistence.py    # SQLite/Postgres adapters
|-- service.py        # FastAPI service wrapper
|-- streaming.py      # Kafka/Redis/WebSocket adapters
|-- tracing.py        # OpenTelemetry tracing
`-- visualization.py  # Mermaid and GraphML export
```

---

## Benchmark Notes

See [BENCHMARKS.md](BENCHMARKS.md) for current benchmark guidance. The practical knobs are graph size, hypervector dimension, snapshot frequency, and rule/path length. For early releases, benchmark results should be treated as environment-specific until a repeatable benchmark harness is added.

---

## Development

```bash
pip install -e ".[dev]"
ruff check .
mypy topologist
pytest -q
python -m build
twine check dist/*
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).

---

## License

MIT
