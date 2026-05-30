from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_project_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', text, flags=re.MULTILINE)
    if match is None:
        raise SystemExit("Could not find project version in pyproject.toml")
    return match.group(1)


def require_contains(path: Path, pattern: str, expected: str) -> None:
    text = path.read_text(encoding="utf-8")
    match = re.search(pattern, text)
    if match is None:
        raise SystemExit(f"Could not find version marker in {path.relative_to(ROOT)}")
    actual = match.group(1)
    if actual != expected:
        raise SystemExit(
            f"Version mismatch in {path.relative_to(ROOT)}: expected {expected}, found {actual}"
        )


def run_install_check() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"],
        cwd=ROOT,
        check=True,
    )


def main() -> int:
    version = read_project_version()
    require_contains(
        ROOT / "topologist" / "__init__.py",
        r'__version__ = "([^"]+)"',
        version,
    )
    require_contains(
        ROOT / "tests" / "test_extensions.py",
        r'assert topologist\.__version__ == "([^"]+)"',
        version,
    )

    if os.environ.get("TOPOLOGIST_PRECOMMIT_INSTALL") == "1":
        run_install_check()

    print(f"Topologist pre-commit check passed for version {version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
