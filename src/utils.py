from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "configs/config.json") -> dict[str, Any]:
    config_path = PROJECT_ROOT / path
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_path(path: str | Path) -> Path:
    resolved = PROJECT_ROOT / path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
