# Topologist

<p align="center">
  <img width="300" height="300" alt="" src="https://github.com/Arkay92/Topologist/raw/main/assets/Topologist.png" />
</p>

A production-hardened **hyperdimensional neuro-symbolic topology system** in Python.

Topologist combines:

- **Hyperdimensional Computing / Vector Symbolic Architecture** for robust distributed representations.
- **Neuro-symbolic graph topology** using NetworkX.
- **Rule-based inference** over symbolic relations.
- **Topology analytics** including PageRank centrality, communities, shortest paths, drift, and anomaly scoring.
- **Persistence and export** to JSON, GraphML, and Mermaid.
- **CLI tooling** for demos and inspection.

---

## Why this exists

Most symbolic graph systems are explainable but brittle. Most neural/vector systems are robust but opaque.

Topologist sits between the two:

```text
Symbolic entities and relations
        ↓
Hyperdimensional encoding
        ↓
Topology graph
        ↓
Reasoning + analytics + anomaly detection
```

Each node and relation is stored symbolically, but also encoded into a high-dimensional bipolar hypervector. This gives you a graph that is queryable and explainable while also having a distributed topology-level memory state.

---

## Install

```bash
pip install topologist
```

For development without installing, ensure the package is in the Python path:

```bash
pip install -e .
python examples/demo.py
```

---

## Quick start

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

Streaming example

```bash
# Run the streaming demo which ingests events, applies inference,
# snapshots state, computes drift, and scores anomalies.
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

## Main features

### 1. Hyperdimensional item memory

Stable symbols are encoded into bipolar vectors:

```text
symbol → {-1, +1}^D
```

The engine supports:

- Binding: elementwise multiplication
- Bundling: majority superposition
- Permutation: cyclic shifts for order/role encoding
- Similarity: cosine similarity

### 2. Symbolic topology graph

The graph is a `networkx.MultiDiGraph`, so it supports multiple relation types between the same source and target.

Example:

```text
HDC --models--> Memory
HDC --enhances--> KnowledgeGraph
KnowledgeGraph --supports--> Reasoning
```

### 3. Rule-based inference

Rules operate over two-hop motifs:

```text
A --relation_a--> B
B --relation_b--> C
----------------------
A --inferred_relation--> C
```

### 4. Drift detection

The global graph state is bundled into a single hypervector snapshot.

```python
system.update_global_state(take_snapshot=True)
drift = system.topology_drift()
```

This lets you measure how much the topology has changed over time.

### 5. Anomaly scoring

Candidate relations can be compared against local topology and relation prototypes using unbinding:

```python
score = system.relation_anomaly_score("A", "unexpected_relation", "B")
```

The scoring method:

1. **Unbinds** the candidate relation to recover the relation signal
2. Compares against the **expected relation hypervector**
3. Checks similarity to **local topology edges** (source outgoing, target incoming)
4. Returns anomaly in [0, 1], where 1 is most anomalous

Higher scores indicate relations that deviate from learned patterns.

### 6. Confidence decay

Knowledge that is not reinforced can gradually lose confidence:

```python
system.decay_confidence()
```

This is useful for agent memory, dynamic knowledge graphs, cybersecurity events, medical evidence tracking, and live topology streams.

---

## Project structure

```text
topologist/
├── __init__.py
├── agent.py                # Agent memory adapters (Claude, OpenClaw, local LLM)
├── ann.py                  # Approximate nearest-neighbor search
├── bridges.py              # PyTorch Geometric bridge for GNN integration
├── cli.py                  # CLI commands (demo, inspect, mermaid export)
├── config.py               # Runtime configuration
├── dsl.py                  # Rule DSL for multi-hop inference
├── engine.py               # Core topology engine
├── exceptions.py           # Custom exception types
├── hdc.py                  # Hyperdimensional computing & vector operations
├── io.py                   # CSV load/save utilities
├── models.py               # Pydantic models (Node, Edge, Rule records)
├── persistence.py          # SQLite/Postgres persistence adapters
├── service.py              # FastAPI service wrapper
├── streaming.py            # Kafka/Redis Streams/WebSocket adapters
├── tracing.py              # OpenTelemetry tracing support
└── visualization.py        # Mermaid & GraphML export
```

---

## Run tests

```bash
pytest -q
```

---

## Production hardening included

This package includes:

- Typed modules
- Pydantic validation
- Custom exceptions
- Save/load roundtrip support
- CLI entrypoint
- Config object
- Test suite
- Export helpers
- No notebook-only assumptions
- No hidden API dependency
- Deterministic seed support
- Dimension validation
- Confidence decay
- Snapshot capping

---

## License

MIT
