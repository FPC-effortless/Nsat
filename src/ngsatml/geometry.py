from __future__ import annotations

import json
from pathlib import Path

from shapely.geometry import shape


def state_bboxes(path: str | Path) -> dict[str, tuple[float, float, float, float]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for f in data.get("features", []):
        props = f.get("properties", {})
        name = str(props.get("shapeName") or props.get("name") or "")
        if not name:
            continue
        geom = shape(f["geometry"])
        out[name] = tuple(float(v) for v in geom.bounds)
    return out
