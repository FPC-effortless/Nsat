from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping")
    required = {"states", "start_date", "end_date"}
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    if not isinstance(data["states"], list):
        raise ValueError("states must be a list; use [] for nationwide dataset builds")
    return data
