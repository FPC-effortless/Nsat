from __future__ import annotations

from io import BytesIO

import pandas as pd
import requests

from .nbs import month_from_text, normalize_state

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


def parse_mirror_cohd(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"state", "cohd_average"}.issubset(frame.columns):
        return pd.DataFrame()
    rows = frame.copy()
    rows["state"] = rows["state"].map(normalize_state)
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
    rows["source_workbook"] = rows.get("source_resource")
    keep = [
        "state", "month", "cohd_ngn_person_day", "source_sheet", "source_workbook",
        "source_resource", "source_resource_id", "source_url", "retrieved_at", "mirror_repo", "transport",
    ]
    for col in keep:
        if col not in rows:
            rows[col] = None
    rows = rows[keep]
    good_months = rows.groupby("month")["state"].nunique()
    good_months = set(good_months[good_months >= 30].index)
    return rows[rows["month"].isin(good_months)].drop_duplicates(["state", "month", "source_resource_id"]).reset_index(drop=True)


def parse_mirror_cpi(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse State CPI Table-5 from the provenance-preserving Parquet mirror.

    NBS State CPI Table-5 is laid out as reference-year Food/All Items,
    previous-month Food/All Items, report-month Food/All Items, annual change,
    then monthly change. The mirror makes duplicate raw Food headers unique as
    food, food_2, food_3, food_4, food_5. Therefore only food_2 and food_3 are
    index levels useful here; food_4 and food_5 are percentages and must never
    be treated as CPI levels.
    """
    if "state" not in frame.columns or "food_2" not in frame.columns or "food_3" not in frame.columns:
        return pd.DataFrame()
    rows = frame.copy()
    rows["state"] = rows["state"].map(normalize_state)
    rows = rows[rows["state"].notna()].copy()
    if rows.empty:
        return pd.DataFrame()
    if "source_sheet" in rows:
        counts = rows.groupby("source_sheet")["state"].nunique()
        good_sheets = set(counts[counts >= 30].index)
        rows = rows[rows["source_sheet"].isin(good_sheets)].copy()
    if rows.empty:
        return pd.DataFrame()

    rows["report_month"] = rows.apply(_source_month, axis=1)
    rows["previous_food_cpi"] = pd.to_numeric(rows["food_2"], errors="coerce")
    rows["report_food_cpi"] = pd.to_numeric(rows["food_3"], errors="coerce")
    rows = rows[rows["report_month"].notna()].copy()
    if rows.empty:
        return pd.DataFrame()

    common = [
        "state", "source_sheet", "source_resource", "source_resource_id", "source_url",
        "retrieved_at", "mirror_repo", "transport",
    ]
    for col in common:
        if col not in rows:
            rows[col] = None

    current = rows[common].copy()
    current["month"] = rows["report_month"]
    current["food_cpi"] = rows["report_food_cpi"]
    previous = rows[common].copy()
    previous["month"] = rows["report_month"] - pd.DateOffset(months=1)
    previous["food_cpi"] = rows["previous_food_cpi"]
    out = pd.concat([previous, current], ignore_index=True)
    out = out[out["food_cpi"].gt(0)].copy()
    out["index_regime"] = out["month"].map(
        lambda m: "2024-base" if m >= pd.Timestamp("2025-01-01") else "2009-11-base"
    )
    out["source_workbook"] = out["source_resource"]
    good = out.groupby(["month", "index_regime"])["state"].nunique()
    good = {key for key, count in good.items() if count >= 30}
    mask = out.apply(lambda r: (r["month"], r["index_regime"]) in good, axis=1)
    return out[mask].drop_duplicates(
        ["state", "month", "index_regime", "source_resource_id"]
    ).reset_index(drop=True)


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
