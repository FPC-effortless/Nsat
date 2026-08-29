from __future__ import annotations

import math
import re
from io import BytesIO
from typing import Iterable

import numpy as np
import openpyxl
import pandas as pd

from .nbs import normalize_state

_MONTH_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2})\b",
    re.I,
)
_MONTH_NUM = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _month(value: object) -> pd.Timestamp | None:
    if value is None:
        return None
    m = _MONTH_RE.search(str(value))
    if not m:
        return None
    return pd.Timestamp(year=int(m.group(2)), month=_MONTH_NUM[m.group(1).casefold()], day=1)


def _num(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.number)):
        x = float(value)
        return x if math.isfinite(x) else None
    text = str(value).replace(",", "").strip()
    try:
        x = float(text)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def _sheet_rows(ws, *, max_rows: int = 140, max_cols: int = 60) -> list[list[object]]:
    rows: list[list[object]] = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        rows.append(list(row[:max_cols]))
        if i + 1 >= max_rows:
            break
    return rows


def parse_state_food_cpi_workbook(
    payload: bytes,
    *,
    source_url: str,
    workbook_name: str,
    index_regime: str = "2009-base",
) -> pd.DataFrame:
    """Parse the all-state Food CPI table from an official NBS workbook.

    Legacy NBS CPI workbooks use a ``State CPI`` sheet with rolling month blocks.
    Each month block contains FOOD and ALL ITEMS values. The parser reconstructs
    merged month headers by forward filling and uses only raw index columns—not
    annual/monthly change columns.
    """
    wb = openpyxl.load_workbook(BytesIO(payload), read_only=True, data_only=True)
    candidates = sorted(
        wb.worksheets,
        key=lambda ws: ("state cpi" not in ws.title.casefold(), ws.title.casefold()),
    )
    records: list[dict[str, object]] = []

    for ws in candidates:
        rows = _sheet_rows(ws)
        if not rows:
            continue
        width = max((len(r) for r in rows), default=0)
        best_state_col: int | None = None
        best_state_rows: list[tuple[int, str]] = []
        for col in range(width):
            found: list[tuple[int, str]] = []
            for ridx, row in enumerate(rows):
                if col < len(row):
                    state = normalize_state(row[col])
                    if state is not None:
                        found.append((ridx, state))
            if len({s for _, s in found}) > len({s for _, s in best_state_rows}):
                best_state_col, best_state_rows = col, found
        if best_state_col is None or len({s for _, s in best_state_rows}) < 30:
            continue

        first_state = min(r for r, _ in best_state_rows)
        header_rows = rows[max(0, first_state - 7):first_state]
        months_by_col: dict[int, pd.Timestamp] = {}
        last_month: pd.Timestamp | None = None
        for col in range(width):
            direct = None
            for header in header_rows:
                if col < len(header):
                    direct = _month(header[col]) or direct
            if direct is not None:
                last_month = direct
            if last_month is not None:
                months_by_col[col] = last_month

        for col, month in months_by_col.items():
            header_stack = " ".join(
                str(header[col]) for header in header_rows if col < len(header) and header[col] not in (None, "")
            ).casefold()
            if "food" not in header_stack:
                continue
            if any(term in header_stack for term in ("annual change", "monthly change", "% change", "year on year", "month on month")):
                continue
            values: list[tuple[str, float]] = []
            for ridx, state in best_state_rows:
                if ridx < len(rows) and col < len(rows[ridx]):
                    x = _num(rows[ridx][col])
                    if x is not None and x > 0:
                        values.append((state, x))
            if len({s for s, _ in values}) < 30:
                continue
            for state, food_index in values:
                records.append({
                    "state": state,
                    "month": month,
                    "food_cpi": food_index,
                    "index_regime": index_regime,
                    "source_sheet": ws.title,
                    "source_workbook": workbook_name,
                    "source_url": source_url,
                })
        if records:
            break

    if not records:
        return pd.DataFrame(columns=[
            "state", "month", "food_cpi", "index_regime", "source_sheet",
            "source_workbook", "source_url",
        ])
    out = pd.DataFrame(records)
    return (
        out.groupby(["state", "month", "index_regime"], as_index=False)
        .agg(
            food_cpi=("food_cpi", "median"),
            source_sheet=("source_sheet", "first"),
            source_workbook=("source_workbook", "first"),
            source_url=("source_url", "first"),
        )
        .sort_values(["month", "state"])
        .reset_index(drop=True)
    )


def consolidate_cpi(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    parts = [f for f in frames if f is not None and not f.empty]
    if not parts:
        return pd.DataFrame()
    all_rows = pd.concat(parts, ignore_index=True)
    # Rolling NBS workbooks repeat prior months. Median should equal the published
    # value; source_count makes disagreements visible instead of silently hiding them.
    grouped = (
        all_rows.groupby(["state", "month", "index_regime"], as_index=False)
        .agg(
            food_cpi=("food_cpi", "median"),
            source_count=("food_cpi", "size"),
            source_min=("food_cpi", "min"),
            source_max=("food_cpi", "max"),
            source_workbooks=("source_workbook", lambda s: "|".join(sorted(set(map(str, s))))),
            source_urls=("source_url", lambda s: "|".join(sorted(set(map(str, s))))),
        )
    )
    grouped["source_disagreement"] = (grouped["source_max"] - grouped["source_min"]).abs()
    return grouped.sort_values(["month", "state"]).reset_index(drop=True)


def add_cpi_targets(frame: pd.DataFrame, *, lags: tuple[int, ...] = (1, 2, 3, 6)) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    keys = ["state", "index_regime"]
    out = frame.copy()
    lookup = out[keys + ["month", "food_cpi"]].drop_duplicates(keys + ["month"])
    for lag in lags:
        prior = lookup.copy()
        prior["month"] = prior["month"] + pd.DateOffset(months=lag)
        prior = prior.rename(columns={"food_cpi": f"food_cpi_lag_{lag}m"})
        out = out.merge(prior, on=keys + ["month"], how="left")
    future = lookup.copy()
    future["month"] = future["month"] - pd.DateOffset(months=1)
    future = future.rename(columns={"food_cpi": "target_food_cpi_1m"})
    out = out.merge(future, on=keys + ["month"], how="left")
    out["target_month"] = out["month"] + pd.DateOffset(months=1)
    out["target_food_cpi_log_change_1m"] = np.where(
        (out["food_cpi"] > 0) & (out["target_food_cpi_1m"] > 0),
        np.log(out["target_food_cpi_1m"] / out["food_cpi"]),
        np.nan,
    )
    out["target_food_cpi_change_1m_pct"] = np.where(
        out["food_cpi"] > 0,
        (out["target_food_cpi_1m"] - out["food_cpi"]) / out["food_cpi"],
        np.nan,
    )
    out["year"] = out["month"].dt.year.astype("int16")
    out["month_number"] = out["month"].dt.month.astype("int8")
    angle = 2.0 * math.pi * (out["month_number"].astype(float) - 1.0) / 12.0
    out["month_sin"] = np.sin(angle)
    out["month_cos"] = np.cos(angle)
    lag_cols = [f"food_cpi_lag_{lag}m" for lag in lags]
    out["lag_feature_count"] = out[lag_cols].notna().sum(axis=1).astype("int8")
    return out
