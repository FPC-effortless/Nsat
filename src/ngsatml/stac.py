from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from pystac_client import Client
else:
    Client = Any

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"


@dataclass(frozen=True)
class CatalogQuery:
    collection: str
    bbox: tuple[float, float, float, float]
    start: str
    end: str
    cloud_cover_max: float | None = None
    limit: int = 500


def search_items(query: CatalogQuery, client: Client | None = None) -> list[dict]:
    if client is None:
        try:
            from pystac_client import Client as PystacClient
        except ImportError as exc:
            raise RuntimeError("pystac-client is required for live STAC queries; install the package dependencies") from exc
        c = PystacClient.open(EARTH_SEARCH)
    else:
        c = client
    kwargs = {
        "collections": [query.collection],
        "bbox": list(query.bbox),
        "datetime": f"{query.start}/{query.end}",
        "max_items": query.limit,
    }
    if query.cloud_cover_max is not None:
        kwargs["query"] = {"eo:cloud_cover": {"lt": query.cloud_cover_max}}
    search = c.search(**kwargs)
    return [item.to_dict() for item in search.items()]


def month_windows(start: str, end: str) -> Iterable[tuple[str, str]]:
    y, m, _ = map(int, start.split("-"))
    ey, em, _ = map(int, end.split("-"))
    while (y, m) < (ey, em):
        start_d = date(y, m, 1)
        if m == 12:
            ny, nm = y + 1, 1
        else:
            ny, nm = y, m + 1
        end_d = date(ny, nm, 1)
        yield start_d.isoformat(), end_d.isoformat()
        y, m = ny, nm


def compact_item(item: dict) -> dict:
    props = item.get("properties", {})
    assets = item.get("assets", {})
    return {
        "id": item.get("id"),
        "collection": item.get("collection"),
        "datetime": props.get("datetime"),
        "cloud_cover": props.get("eo:cloud_cover"),
        "bbox": item.get("bbox"),
        "assets": {k: v.get("href") for k, v in assets.items() if isinstance(v, dict) and v.get("href")},
    }


def write_manifest(items: list[dict], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([compact_item(i) for i in items], indent=2), encoding="utf-8")
    return p
