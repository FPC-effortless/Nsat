from __future__ import annotations

import json
from pathlib import Path

import requests

GEOB_API = "https://www.geoboundaries.org/api/current/gbOpen/NGA/ADM1/"


def fetch_boundary_metadata(session: requests.Session | None = None) -> dict:
    s = session or requests.Session()
    r = s.get(GEOB_API, timeout=60)
    r.raise_for_status()
    return r.json()


def download_adm1(output_dir: str | Path, session: requests.Session | None = None) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    s = session or requests.Session()
    metadata = fetch_boundary_metadata(s)
    if metadata.get("boundaryLicense") != "Creative Commons Attribution 4.0 International (CC BY 4.0)":
        raise RuntimeError("Unexpected geoBoundaries license; review before proceeding")
    url = metadata["gjDownloadURL"]
    r = s.get(url, timeout=120)
    r.raise_for_status()
    target = out / "nga_adm1.geojson"
    target.write_bytes(r.content)
    (out / "nga_adm1.metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return target


def filter_states_geojson(source: str | Path, states: list[str], target: str | Path) -> Path:
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    wanted = {s.casefold() for s in states}
    kept = []
    observed = set()
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        name = str(props.get("shapeName") or props.get("name") or "")
        if name.casefold() in wanted:
            kept.append(feature)
            observed.add(name.casefold())
    missing = sorted(wanted - observed)
    if missing:
        raise ValueError(f"States not found in boundary file: {missing}")
    output = {"type": "FeatureCollection", "features": kept}
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output), encoding="utf-8")
    return path
