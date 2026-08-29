from __future__ import annotations

import html as html_lib
import math
import re
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import numpy as np
import openpyxl
import pandas as pd
import requests

CATALOG_URL = "https://microdata.nigerianstat.gov.ng/index.php/catalog/162/related-materials"
ELIBRARY_URL = "https://www.nigerianstat.gov.ng/elibrary/"

NIGERIA_STATES = [
    "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue", "Borno",
    "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti", "Enugu", "FCT", "Gombe", "Imo",
    "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Kogi", "Kwara", "Lagos", "Nasarawa",
    "Niger", "Ogun", "Ondo", "Osun", "Oyo", "Plateau", "Rivers", "Sokoto", "Taraba",
    "Yobe", "Zamfara",
]

_STATE_ALIASES = {
    "abuja": "FCT",
    "fct": "FCT",
    "fct abuja": "FCT",
    "abuja fct": "FCT",
    "federal capital territory": "FCT",
    "akwa ibom": "Akwa Ibom",
    "akwaibom": "Akwa Ibom",
    "cross river": "Cross River",
    "crossriver": "Cross River",
    "nasarawa": "Nasarawa",
    "nassarawa": "Nasarawa",
}
for _state in NIGERIA_STATES:
    _STATE_ALIASES.setdefault(_state.casefold(), _state)

_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

_WFP_MAP = [
    (re.compile(r"rice.*local|local.*rice", re.I), "Rice (local)"),
    (re.compile(r"rice.*import", re.I), "Rice (imported)"),
    (re.compile(r"beans?.*brown", re.I), "Cowpeas (brown)"),
    (re.compile(r"maize.*white|white.*maize", re.I), "Maize (white)"),
    (re.compile(r"maize.*yellow|yellow.*maize", re.I), "Maize (yellow)"),
    (re.compile(r"\bmaize\b", re.I), "Maize"),
    (re.compile(r"\btomato", re.I), "Tomatoes"),
    (re.compile(r"\bonion", re.I), "Onions"),
    (re.compile(r"yam.*tuber|\byam\b", re.I), "Yam"),
    (re.compile(r"beef.*boneless|boneless.*beef", re.I), "Meat (beef)"),
    (re.compile(r"garri.*white|gari.*white", re.I), "Gari (white)"),
    (re.compile(r"garri.*yellow|gari.*yellow", re.I), "Cassava meal (gari, yellow)"),
    (re.compile(r"palm.*oil|oil.*palm", re.I), "Oil (palm)"),
    (re.compile(r"vegetable.*oil|oil.*vegetable", re.I), "Oil (vegetable)"),
    (re.compile(r"\begg", re.I), "Eggs"),
    (re.compile(r"milk.*powder|powder.*milk", re.I), "Milk (powder)"),
]


@dataclass(frozen=True)
class NBSResource:
    title: str
    url: str
    month: pd.Timestamp | None = None
    source_page: str | None = None
    source: str = "nbs"


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    text = html_lib.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_state(value: object) -> str | None:
    text = _clean_text(value).casefold()
    text = re.sub(r"\bstate\b", "", text)
    text = re.sub(r"[^a-z ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _STATE_ALIASES.get(text)


def month_from_text(text: str) -> pd.Timestamp | None:
    cleaned = _clean_text(text).casefold().replace("_", " ").replace("-", " ")
    year_match = re.search(r"\b(20\d{2})\b", cleaned)
    if not year_match:
        return None
    year = int(year_match.group(1))
    tokens = re.findall(r"[a-z]+", cleaned)
    for token in tokens:
        if token in _MONTHS:
            return pd.Timestamp(year=year, month=_MONTHS[token], day=1)
    return None


def canonical_wfp_commodity(item: str) -> str | None:
    for pattern, value in _WFP_MAP:
        if pattern.search(item):
            return value
    return None


def parse_item_unit(item: str) -> tuple[str, float, str]:
    """Return canonical unit, package quantity in that unit, and family."""
    text = item.casefold().replace(",", " ")
    match = re.search(r"(\d+(?:\.\d+)?)\s*(kg|kilograms?|g|grams?|litres?|liters?|ltr|l\b|pieces?|pcs\b|tubers?)", text)
    if match:
        qty = float(match.group(1))
        unit = match.group(2)
        if unit in {"kg", "kilogram", "kilograms"}:
            return "kg", qty, "mass"
        if unit in {"g", "gram", "grams"}:
            return "kg", qty / 1000.0, "mass"
        if unit in {"litre", "litres", "liter", "liters", "ltr", "l"}:
            return "l", qty, "volume"
        return "item", qty, "count"
    if re.search(r"\bone\s+bottle\b|\b1\s*bottle\b", text):
        return "bottle", 1.0, "package"
    if "bottle" in text:
        return "bottle", 1.0, "package"
    return "package", 1.0, "package"


def discover_downloads(session: requests.Session | None = None) -> list[NBSResource]:
    """Discover NBS NADA resources, best-effort."""
    s = session or requests.Session()
    r = s.get(CATALOG_URL, timeout=(15, 30))
    r.raise_for_status()
    html = r.text
    found: list[NBSResource] = []
    pattern = re.compile(r'href=["\']([^"\']*/catalog/162/download/\d+)["\']', re.I)
    for m in pattern.finditer(html):
        url = urljoin(CATALOG_URL, m.group(1))
        context = _clean_text(html[max(0, m.start() - 700):m.start() + 100])
        month = month_from_text(context)
        found.append(NBSResource(title=context[-240:] or "NBS resource", url=url, month=month, source_page=CATALOG_URL, source="nada"))
    dedup: dict[str, NBSResource] = {}
    for item in found:
        dedup.setdefault(item.url, item)
    return list(dedup.values())


def discover_elibrary_table_downloads(
    *,
    start_date: str | pd.Timestamp = "2022-01-01",
    end_date: str | pd.Timestamp | None = None,
    session: requests.Session | None = None,
) -> list[NBSResource]:
    """Discover direct e-library table workbooks without assuming filenames."""
    s = session or requests.Session()
    response = s.get(ELIBRARY_URL, timeout=(15, 30))
    response.raise_for_status()
    body = response.text
    start = pd.Timestamp(start_date).to_period("M").to_timestamp()
    end = pd.Timestamp(end_date).to_period("M").to_timestamp() if end_date is not None else None
    report_links: list[tuple[str, pd.Timestamp]] = []
    for href, label in re.findall(r'href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', body, flags=re.I | re.S):
        text = _clean_text(label)
        month = month_from_text(text)
        if month is None or "food" not in text.casefold() or "price" not in text.casefold():
            continue
        if month < start or (end is not None and month >= end):
            continue
        report_links.append((urljoin(ELIBRARY_URL, href), month))

    results: dict[str, NBSResource] = {}
    for page_url, month in report_links:
        try:
            page = s.get(page_url, timeout=(15, 30))
            page.raise_for_status()
        except requests.RequestException:
            continue
        for href, label in re.findall(r'href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', page.text, flags=re.I | re.S):
            text = _clean_text(label)
            absolute = urljoin(page_url, href)
            if absolute.lower().endswith((".xlsx", ".xls")) or "download table" in text.casefold():
                results.setdefault(absolute, NBSResource(title=text or f"NBS table {month:%Y-%m}", url=absolute, month=month, source_page=page_url, source="elibrary"))
    return list(results.values())


def workbook_payloads(content: bytes, fallback_name: str = "download.xlsx") -> list[tuple[str, bytes]]:
    if zipfile.is_zipfile(BytesIO(content)):
        with zipfile.ZipFile(BytesIO(content)) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/") and n.lower().endswith((".xlsx", ".xlsm", ".xls"))]
            if names:
                return [(Path(n).name, zf.read(n)) for n in names]
    return [(fallback_name, content)]


def _find_header_row(raw: pd.DataFrame, required_any: Iterable[str]) -> int | None:
    needles = {x.casefold() for x in required_any}
    for idx, row in raw.iterrows():
        values = {_clean_text(v).casefold() for v in row.tolist() if _clean_text(v)}
        if values & needles:
            return int(idx)
    return None


def parse_table_workbook(content: bytes, resource: NBSResource) -> pd.DataFrame:
    """Conservatively parse Selected Food Price Watch workbooks.

    These tables are useful for national/zonal commodity validation. They are
    not assumed to contain a full state x commodity matrix.
    """
    frames: list[pd.DataFrame] = []
    for name, payload in workbook_payloads(content):
        if not payload.startswith(b"PK"):
            continue
        wb = openpyxl.load_workbook(BytesIO(payload), read_only=True, data_only=True)
        for ws in wb.worksheets:
            values = list(ws.values)
            if not values:
                continue
            raw = pd.DataFrame(values)
            header_idx = _find_header_row(raw.head(30), {"item", "items", "commodity", "food item"})
            if header_idx is None:
                continue
            header = [_clean_text(v) or f"column_{i}" for i, v in enumerate(raw.iloc[header_idx].tolist())]
            table = raw.iloc[header_idx + 1:].copy()
            table.columns = header
            table = table.dropna(how="all")
            table["source_workbook"] = name
            table["source_sheet"] = ws.title
            table["source_url"] = resource.url
            table["report_month"] = resource.month
            frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_resource_index(resources: list[NBSResource], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {
            "title": r.title,
            "url": r.url,
            "month": None if r.month is None else r.month.strftime("%Y-%m-%d"),
            "source_page": r.source_page,
            "source": r.source,
        }
        for r in resources
    ]).to_csv(path, index=False)
    return path
