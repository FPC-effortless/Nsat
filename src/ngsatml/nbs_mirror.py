from __future__ import annotations

import re
from io import BytesIO
from typing import Iterable

import pandas as pd
import requests

from .nbs import NIGERIA_STATES, month_from_text, normalize_state

HF_AUTHOR = "electricsheepafrica"
HF_DATASETS_API = "https://huggingface.co/api/datasets"
HF_PARQUET_API = "https://datasets-server.huggingface.co/parquet"
CPI_SEARCH = "africa-nigeria-consumer-price-index-and-inflation"
COHD_SEARCH = "africa-nigeria-cost-of-healthy-diet"


def discover_mirror_repos(search: str, *, session: requests.Session | None = None) -> list[str]:
    s = session or requests.Session()
    r = s.get(
        HF_DATASETS_API,
        params={"author": HF_AUTHOR, "search": search, "limit": 100, "full": "true"},
        timeout=(10, 30),
    )
    r.raise_for_status()
    return sorted({row["id"] for row in r.json() if row.get("id")})


def load_mirror_repo(repo: str, *, session: requests.Session | None = None) -> pd.DataFrame:
    s = session or requests.Session()
    meta = s.get(HF_PARQUET_API, params={"dataset": repo}, timeout=(10, 30))
    meta.raise_for_status()
    files = meta.json().get("parquet_files", [])
    if not files:
        raise RuntimeError(f"No parquet files exposed for {repo}")
    parts: list[pd.DataFrame] = []
    for item in files:
        response = s.get(item["url"], timeout=(10, 60))
        response.raise_for_status()
        parts.append(pd.read_parquet(BytesIO(response.content)))
    frame = pd.concat(parts, ignore_index=True)
    frame["mirror_repo"] = repo
    frame["transport"] = "huggingface-mirror"
    return frame


def _source_month(row: pd.Series) -> pd.Timestamp | None:
    for key in ("source_sheet", "source_resource"):
        value = row.get(key)
        if pd.notna(value):
            month = month_from_text(str(value))
            if month is not None:
                return month
    return None


def _canonical_state_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_state)


def parse_mirror_cohd(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"state", "cohd_average"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    rows = frame.copy()
    rows["state"] = _canonical_state_series(rows["state"])
    rows = rows[rows["state"].notna()].copy()
    if "source_sheet" in rows:
        preferred = rows[rows["source_sheet"].astype(str).str.contains("national average", case=False, na=False)]
        if not preferred.empty:
            rows = preferred
    rows["month"] = rows.apply(_source_month, axis=1)
    rows["cohd_ngn_person_day"] = pd.to_numeric(rows["cohd_average"], errors="coerce")
    rows = rows[rows["month"].notna() & rows["cohd_ngn_person_day"].gt(0)].copy()
    if rows.empty:
        return pd.DataFrame()
    keep = [
        "state", "month", "cohd_ngn_person_day", "source_sheet", "source_resource",
        "source_resource_id", "source_url", "retrieved_at", "mirror_repo", "transport",
    ]
    for col in keep:
        if col not in rows:
            rows[col] = None
    rows = rows[keep]
    # Reject sheets that are not genuine all-state tables.
    good_months = rows.groupby("month")["state"].nunique()
    good_months = set(good_months[good_months >= 30].index)
    return rows[rows["month"].isin(good_months)].drop_duplicates(["state", "month", "source_resource_id"]).reset_index(drop=True)


def _food_columns(frame: pd.DataFrame) -> list[str]:
    # Electric Sheep preserves source-column order while making duplicate Food
    # headers unique as food, food_2, ... . In NBS State CPI Table-5 the last
    # raw Food column is the report month; annual/monthly-change headers have
    # distinct names and therefore are not included here.
    pattern = re.compile(r"^food(?:_(\d+))?$", re.I)
    return [c for c in frame.columns if pattern.match(str(c))]


def parse_mirror_cpi(frame: pd.DataFrame) -> pd.DataFrame:
    if "state" not in frame.columns:
        return pd.DataFrame()
    food_cols = _food_columns(frame)
    if not food_cols:
        return pd.DataFrame()
    rows = frame.copy()
    rows["state"] = _canonical_state_series(rows["state"])
    rows = rows[rows["state"].notna()].copy()
    if rows.empty:
        return pd.DataFrame()

    # Select only source sheets that contain near-complete state coverage.
    if "source_sheet" in rows:
        counts = rows.groupby("source_sheet")["state"].nunique()
        good_sheets = set(counts[counts >= 30].index)
        rows = rows[rows["source_sheet"].isin(good_sheets)].copy()
    if rows.empty:
        return pd.DataFrame()

    rows["month"] = rows.apply(_source_month, axis=1)
    values = rows[food_cols].apply(pd.to_numeric, errors="coerce")
    # Last non-null Food field in source order = report-month Food index.
    rows["food_cpi"] = values.ffill(axis=1).iloc[:, -1]
    rows = rows[rows["month"].notna() & rows["food_cpi"].gt(0)].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["index_regime"] = rows["month"].map(lambda m: "2024-base" if m >= pd.Timestamp("2025-01-01") else "2009-11-base")
    keep = [
        "state", "month", "food_cpi", "index_regime", "source_sheet", "source_resource",
        "source_resource_id", "source_url", "retrieved_at", "mirror_repo", "transport",
    ]
    for col in keep:
        if col not in rows:
            rows[col] = None
    rows = rows[keep]
    good_months = rows.groupby(["month", "index_regime"])["state"].nunique()
    good_months = {key for key, count in good_months.items() if count >= 30}
    mask = rows.apply(lambda r: (r["month"], r["index_regime"]) in good_months, axis=1)
    return rows[mask].drop_duplicates(["state", "month", "index_regime", "source_resource_id"]).reset_index(drop=True)


def load_all_mirror_targets(
    *,
    session: requests.Session | None = None,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[dict[str, str]], list[str]]:
    """Return CPI frames, CoHD frames, transport errors, and discovered repos."""
    s = session or requests.Session()
    errors: list[dict[str, str]] = []
    repos: list[str] = []
    cpi_frames: list[pd.DataFrame] = []
    cohd_frames: list[pd.DataFrame] = []
    for search, parser, target in (
        (CPI_SEARCH, parse_mirror_cpi, cpi_frames),
        (COHD_SEARCH, parse_mirror_cohd, cohd_frames),
    ):
        try:
            found = discover_mirror_repos(search, session=s)
            repos.extend(found)
        except Exception as exc:
            errors.append({"kind": search, "repo": "<discovery>", "error": f"{type(exc).__name__}: {exc}"})
            continue
        for repo in found:
            try:
                parsed = parser(load_mirror_repo(repo, session=s))
                if not parsed.empty:
                    target.append(parsed)
            except Exception as exc:
                errors.append({"kind": search, "repo": repo, "error": f"{type(exc).__name__}: {exc}"})
    return cpi_frames, cohd_frames, errors, sorted(set(repos))
