# Changelog

All notable changes to Topologist are documented here.

## 0.1.11

- Added focused tests for DSL parsing, persistence, streaming, CLI, and FastAPI service behavior.
- Added package validation to CI with `python -m build` and `twine check dist/*`.
- Switched the publish workflow to PyPI Trusted Publishing.
- Added a cybersecurity worked example with multi-hop inference, anomaly scoring, drift tracking, and Mermaid export.
- Added repository polish docs: contributing guide, benchmark notes, badges, and architecture overview.
- Updated package metadata to the modern SPDX license form.

## 0.1.9

- Corrected package URLs for GitHub, documentation, and issues.
- Loaded `__version__` from installed package metadata with a local fallback.
- Namespaced HDC node and relation encodings.
- Replaced cyclic confidence permutation with quantized confidence bands.
- Improved relation anomaly scoring using unbinding and local topology checks.
- Persisted configuration, HDC item memory, nodes, edges, and snapshots in SQL adapters.
- Fixed Redis stream ID handling.
- Expanded README documentation for production modules.

## 0.1.0

- Initial public package structure with core HDC topology engine, reasoning rules, CLI, JSON persistence, and Mermaid export.
