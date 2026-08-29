from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

CATALOG_URL = "https://microdata.nigerianstat.gov.ng/index.php/catalog/162/related-materials"


@dataclass(frozen=True)
class NBSResource:
    title: str
    url: str


def discover_downloads(session: requests.Session | None = None) -> list[NBSResource]:
    """Discover report/table links conservatively from the NBS catalog HTML."""
    s = session or requests.Session()
    r = s.get(CATALOG_URL, timeout=60)
    r.raise_for_status()
    html = r.text
    found = []
    for m in re.finditer(r'href=["\']([^"\']*/catalog/162/download/\d+)["\']', html, flags=re.I):
        url = urljoin(CATALOG_URL, m.group(1))
        context = re.sub(r"<[^>]+>", " ", html[max(0, m.start()-250):m.start()])
        context = re.sub(r"\s+", " ", context).strip()
        found.append(NBSResource(title=context[-160:] or "NBS resource", url=url))
    seen = set()
    dedup = []
    for item in found:
        if item.url not in seen:
            seen.add(item.url)
            dedup.append(item)
    return dedup


def normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a simple state x commodity price table when columns are unambiguous."""
    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]
    candidates = [c for c in work.columns if c.casefold() in {"state", "states", "state name"}]
    if not candidates:
        raise ValueError("No explicit state column found; manual parser mapping required")
    state_col = candidates[0]
    work = work.rename(columns={state_col: "state"})
    work["state"] = work["state"].astype(str).str.strip()
    return work


def save_resource_index(resources: list[NBSResource], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.__dict__ for r in resources]).to_csv(p, index=False)
    return p
