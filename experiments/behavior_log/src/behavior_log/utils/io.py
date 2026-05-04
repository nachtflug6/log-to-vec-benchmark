"""I/O helpers used across behavior-log scripts."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""

    text = Path(path).read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    return _load_simple_yaml_without_dependency(text)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save JSON with parent directory creation."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_simple_yaml_without_dependency(text: str) -> dict[str, Any]:
    """Parse the small config subset used by this experiment."""

    result: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Unsupported config line: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            result[key] = None
            continue
        lowered = value.lower()
        if lowered == "true":
            result[key] = True
            continue
        if lowered == "false":
            result[key] = False
            continue
        try:
            result[key] = ast.literal_eval(value)
            continue
        except (SyntaxError, ValueError):
            pass
        result[key] = value
    return result
