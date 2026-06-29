from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()
