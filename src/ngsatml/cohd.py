from __future__ import annotations

import math
from io import BytesIO
from typing import Iterable

import numpy as np
import openpyxl
import pandas as pd

from .nbs import normalize_state


def _num(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        x = float(str(value).replace(",", "").replace("₦", "").strip())
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def parse_state_cohd_workbook(
    payload: bytes,
    *,
    month: pd.Timestamp,
    source_url: str,
    workbook_name: str,
) -> pd.DataFrame:
    """Parse the all-state Cost of a Healthy Diet average from an NBS workbook."""
    wb = openpyxl.load_workbook(BytesIO(payload), read_only=True, data_only=True)
    worksheets = sorted(
        wb.worksheets,
        key=lambda ws: (
            "national average" not in ws.title.casefold(),
            "state" not in ws.title.casefold(),
            ws.title.casefold(),
        ),
    )
    records: list[dict[str, object]] = []
    month = pd.Timestamp(month).to_period("M").to_timestamp()

    for ws in worksheets:
        rows: list[list[object]] = []
        for ridx, row in enumerate(ws.iter_rows(values_only=True)):
            rows.append(list(row[:40]))
            if ridx >= 100:
                break
        if not rows:
            continue
        width = max((len(r) for r in rows), default=0)

        state_col: int | None = None
        state_rows: list[tuple[int, str]] = []
        for col in range(width):
            matches: list[tuple[int, str]] = []
            for ridx, row in enumerate(rows):
                if col >= len(row):
                    continue
                state = normalize_state(row[col])
                if state is not None:
                    matches.append((ridx, state))
            if len({s for _, s in matches}) > len({s for _, s in state_rows}):
                state_col = col
                state_rows = matches
        if state_col is None or len({s for _, s in state_rows}) < 30:
            continue

        first_state = min(r for r, _ in state_rows)
        header_rows = rows[max(0, first_state - 5):first_state]
        best: tuple[int, list[tuple[str, float]]] | None = None
        for col in range(width):
            if col == state_col:
                continue
            header = " ".join(
                str(row[col]) for row in header_rows
                if col < len(row) and row[col] not in (None, "")
            ).casefold()
            context = f"{ws.title.casefold()} {header}"
            if "cohd" not in context and "healthy diet" not in context:
                continue
            if "urban" in header or "rural" in header:
                continue
            values: list[tuple[str, float]] = []
            for ridx, state in state_rows:
                if ridx < len(rows) and col < len(rows[ridx]):
                    x = _num(rows[ridx][col])
                    if x is not None:
                        values.append((state, x))
            if len({s for s, _ in values}) >= 30:
                if best is None or len({s for s, _ in values}) > len({s for s, _ in best[1]}):
                    best = (col, values)
        if best is None:
            continue

        for state, value in best[1]:
            records.append({
                "state": state,
                "month": month,
                "cohd_ngn_person_day": value,
                "source_sheet": ws.title,
                "source_workbook": workbook_name,
                "source_url": source_url,
            })
        break

    columns = [
        "state", "month", "cohd_ngn_person_day", "source_sheet",
        "source_workbook", "source_url",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records).drop_duplicates(["state", "month"]).sort_values(["month", "state"]).reset_index(drop=True)


def consolidate_cohd(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    parts = [f for f in frames if f is not None and not f.empty]
    if not parts:
        return pd.DataFrame()
    rows = pd.concat(parts, ignore_index=True)
    out = (
        rows.groupby(["state", "month"], as_index=False)
        .agg(
            cohd_ngn_person_day=("cohd_ngn_person_day", "median"),
            source_count=("cohd_ngn_person_day", "size"),
            source_min=("cohd_ngn_person_day", "min"),
            source_max=("cohd_ngn_person_day", "max"),
            source_workbooks=("source_workbook", lambda s: "|".join(sorted(set(map(str, s))))),
            source_urls=("source_url", lambda s: "|".join(sorted(set(map(str, s))))),
        )
    )
    out["source_disagreement"] = (out["source_max"] - out["source_min"]).abs()
    return out.sort_values(["month", "state"]).reset_index(drop=True)


def add_cohd_targets(frame: pd.DataFrame, *, lags: tuple[int, ...] = (1, 2, 3, 6, 12)) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    lookup = out[["state", "month", "cohd_ngn_person_day"]].drop_duplicates(["state", "month"])
    for lag in lags:
        prior = lookup.copy()
        prior["month"] = prior["month"] + pd.DateOffset(months=lag)
        prior = prior.rename(columns={"cohd_ngn_person_day": f"cohd_lag_{lag}m"})
        out = out.merge(prior, on=["state", "month"], how="left")
    future = lookup.copy()
    future["month"] = future["month"] - pd.DateOffset(months=1)
    future = future.rename(columns={"cohd_ngn_person_day": "target_cohd_1m"})
    out = out.merge(future, on=["state", "month"], how="left")
    out["target_month"] = out["month"] + pd.DateOffset(months=1)
    out["target_cohd_log_change_1m"] = np.where(
        (out["cohd_ngn_person_day"] > 0) & (out["target_cohd_1m"] > 0),
        np.log(out["target_cohd_1m"] / out["cohd_ngn_person_day"]),
        np.nan,
    )
    out["target_cohd_change_1m_pct"] = np.where(
        out["cohd_ngn_person_day"] > 0,
        (out["target_cohd_1m"] - out["cohd_ngn_person_day"]) / out["cohd_ngn_person_day"],
        np.nan,
    )
    out["year"] = out["month"].dt.year.astype("int16")
    out["month_number"] = out["month"].dt.month.astype("int8")
    angle = 2.0 * math.pi * (out["month_number"].astype(float) - 1.0) / 12.0
    out["month_sin"] = np.sin(angle)
    out["month_cos"] = np.cos(angle)
    lag_cols = [f"cohd_lag_{lag}m" for lag in lags]
    out["lag_feature_count"] = out[lag_cols].notna().sum(axis=1).astype("int8")
    return out
