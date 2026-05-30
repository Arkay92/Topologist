# Benchmark Notes

Topologist performance depends mainly on graph size, hypervector dimension, snapshot frequency, and rule/path length.

## Current Expectations

These are qualitative notes rather than published benchmark claims:

- Node and edge insertion are dominated by NetworkX mutation plus HDC vector lookup/binding.
- HDC relation encoding is `O(dim)` for each encoded relation.
- Global state updates bundle all node and edge vectors, so they scale with `(nodes + edges) * dim`.
- Multi-hop inference scales with the number of reachable paths matching the relation sequence.
- Snapshot frequency matters. Taking a snapshot on every event is useful for demos and audit trails, but batch snapshots are usually better for high-throughput streams.
- ANN lookup can use the optional Annoy dependency, but exact fallback search is linear in the number of stored node vectors.

## Suggested Benchmark Matrix

Use this matrix when adding a repeatable benchmark harness:

| Nodes | Edges | Dimension | Rule length | Snapshot mode |
| ---: | ---: | ---: | ---: | --- |
| 100 | 500 | 1,024 | 2 | every event |
| 1,000 | 5,000 | 2,048 | 2 | every 100 events |
| 10,000 | 50,000 | 4,096 | 3 | every 1,000 events |

## Metrics To Capture

- Edge ingestion throughput.
- `update_global_state()` latency.
- Multi-hop inference latency and inferred edge count.
- `relation_anomaly_score()` latency.
- JSON and SQLite save/load time.
- Peak memory usage.

## Reproducibility

Benchmark reports should include:

- Python version.
- Operating system.
- CPU model.
- Dependency versions.
- Hypervector dimension.
- Random seed.
- Graph shape and relation distribution.
