# Contributing

Thanks for working on Topologist. The project is still early, so contributions should favor correctness, tests, and clear examples over broad feature expansion.

## Development Setup

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Checks

Run the same checks CI runs:

```bash
ruff check .
mypy topologist
pytest -q
python -m build
twine check dist/*
```

## Contribution Guidelines

- Keep changes scoped to one concern.
- Add or update tests for user-facing behavior, persistence formats, inference behavior, and service/streaming adapters.
- Prefer existing public APIs and local patterns over new abstractions.
- Keep optional integrations optional. Missing Kafka, Redis, Postgres, FastAPI, tracing, or GNN dependencies should fail with clear install guidance.
- Avoid committing generated runtime outputs such as local topology JSON, Mermaid exports, coverage files, and build artifacts.

## Release Notes

When a change affects users, update `CHANGELOG.md`. For packaging changes, verify both `python -m build` and `twine check dist/*`.

## Trusted Publishing

Publishing is handled by GitHub Actions using PyPI Trusted Publishing. Releases should be created from version tags matching `v*`, after the CI workflow is green.
