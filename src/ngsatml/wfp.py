from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable
import math
import re

import numpy as np
import pandas as pd
import requests

WFP_CSV_URL = (
    "https://data.humdata.org/dataset/42db041f-7aaf-4ab4-961f-2a12096861e7/"
    "resource/12b51155-0cd3-4806-9924-61ede4077591/download/wfp_food_prices_nga.csv"
)

LABEL_KEYS = ["market_id", "commodity", "unit", "pricetype"]
_UNIT_RE = re.compile(r"^\s*(?:(\d+(?:\.\d+)?)\s*)?([A-Za-z]+)\s*$")


def download_wfp_csv(path: str | Path, url: str = WFP_CSV_URL, session: requests.Session | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target
    s = session or requests.Session()
    with s.get(url, timeout=120, stream=True) as response:
        response.raise_for_status()
        with target.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    return target


def read_wfp_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if not df.empty:
        first = df.iloc[0].astype(str)
        if int(first.str.startswith("#").sum()) >= max(3, len(first) // 3):
            df = df.iloc[1:].reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _casefold_set(values: Iterable[str] | None) -> set[str]:
    return {str(v).strip().casefold() for v in (values or []) if str(v).strip()}


def normalize_unit(unit: str) -> tuple[str, float, str]:
    """Return (canonical_unit, quantity_in_canonical_unit, unit_family).

    WFP Nigeria mixes package units such as KG, 500 G, 2.5 KG, 30 pcs and
    100 Tubers. Price targets are only comparable after converting package
    prices to a canonical kg/l/item basis.
    """
    raw = str(unit).strip()
    match = _UNIT_RE.match(raw)
    if not match:
        return f"other:{raw.casefold()}", 1.0, "other"
    quantity = float(match.group(1) or 1.0)
    name = match.group(2).casefold()
    if name in {"kg", "kgs", "kilogram", "kilograms"}:
        return "kg", quantity, "mass"
    if name in {"g", "gram", "grams"}:
        return "kg", quantity / 1000.0, "mass"
    if name in {"l", "lt", "ltr", "litre", "litres", "liter", "liters"}:
        return "l", quantity, "volume"
    if name in {"pc", "pcs", "piece", "pieces", "unit", "units", "tuber", "tubers"}:
        return "item", quantity, "count"
    return f"other:{name}", quantity, "other"


def normalize_prices(
    df: pd.DataFrame,
    *,
    states: Iterable[str] | None = None,
    commodities: Iterable[str] | None = None,
    units: Iterable[str] | None = ("KG",),
    pricetypes: Iterable[str] | None = None,
) -> pd.DataFrame:
    required = {
        "date", "admin1", "admin2", "market", "market_id", "latitude", "longitude",
        "commodity", "unit", "pricetype", "currency", "price",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"WFP table missing required columns: {sorted(missing)}")

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    for col in ["latitude", "longitude", "price", "usdprice", "market_id", "commodity_id"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    for col in ["admin1", "admin2", "market", "commodity", "unit", "pricetype", "currency", "category", "priceflag"]:
        if col in work.columns:
            work[col] = work[col].astype(str).str.strip()

    work = work.dropna(subset=["date", "market_id", "latitude", "longitude", "price"])
    work = work[(work["price"] > 0) & work["latitude"].between(-90, 90) & work["longitude"].between(-180, 180)]
    work = work[work["currency"].str.casefold().eq("ngn")]

    state_filter = _casefold_set(states)
    if state_filter:
        work = work[work["admin1"].str.casefold().isin(state_filter)]
    commodity_filter = _casefold_set(commodities)
    if commodity_filter:
        work = work[work["commodity"].str.casefold().isin(commodity_filter)]
    unit_filter = _casefold_set(units)
    if unit_filter:
        work = work[work["unit"].str.casefold().isin(unit_filter)]
    type_filter = _casefold_set(pricetypes)
    if type_filter:
        work = work[work["pricetype"].str.casefold().isin(type_filter)]

    work["market_id"] = work["market_id"].astype("int64")
    work["month"] = work["date"].dt.to_period("M").dt.to_timestamp()
    parsed = work["unit"].map(normalize_unit)
    work["source_unit"] = work["unit"]
    work["unit"] = parsed.map(lambda x: x[0])
    work["unit_quantity"] = parsed.map(lambda x: x[1])
    work["unit_family"] = parsed.map(lambda x: x[2])
    work = work[work["unit_quantity"].gt(0)]
    work["price_ngn_base"] = work["price"] / work["unit_quantity"]
    if "usdprice" in work.columns:
        work["price_usd_base"] = work["usdprice"] / work["unit_quantity"]

    group_cols = [
        "admin1", "admin2", "market", "market_id", "latitude", "longitude",
        "month", "commodity", "unit", "unit_family", "pricetype", "currency",
    ]
    named_aggs: dict[str, tuple[str, object]] = {
        "price_ngn": ("price_ngn_base", "median"),
        "price_observations": ("price_ngn_base", "count"),
        "source_unit_count": ("source_unit", "nunique"),
        "source_units": ("source_unit", lambda s: "|".join(sorted(set(str(v) for v in s if str(v))))),
    }
    if "price_usd_base" in work.columns:
        named_aggs["price_usd"] = ("price_usd_base", "median")
    if "category" in work.columns:
        named_aggs["category"] = ("category", "first")
    if "commodity_id" in work.columns:
        named_aggs["commodity_id"] = ("commodity_id", "first")
    if "priceflag" in work.columns:
        named_aggs["priceflag"] = ("priceflag", lambda s: "|".join(sorted(set(str(v) for v in s if str(v)))) )

    labels = work.groupby(group_cols, dropna=False).agg(**named_aggs).reset_index()
    return labels.sort_values(["month", "admin1", "market", "commodity", "pricetype"]).reset_index(drop=True)


def add_calendar_targets(labels: pd.DataFrame, lags: tuple[int, ...] = (1, 2, 3, 6, 12)) -> pd.DataFrame:
    required = set(LABEL_KEYS + ["month", "price_ngn"])
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Missing label columns: {sorted(missing)}")
    out = labels.copy()
    lookup = labels[LABEL_KEYS + ["month", "price_ngn"]].copy()

    for lag in lags:
        prior = lookup.copy()
        prior["month"] = prior["month"] + pd.DateOffset(months=lag)
        prior = prior.rename(columns={"price_ngn": f"price_lag_{lag}m"})
        out = out.merge(prior, on=LABEL_KEYS + ["month"], how="left")

    future = lookup.copy()
    future["target_month"] = future["month"] - pd.DateOffset(months=1)
    future = future.drop(columns=["month"]).rename(columns={"price_ngn": "target_price_ngn_1m", "target_month": "month"})
    out = out.merge(future, on=LABEL_KEYS + ["month"], how="left")
    out["target_month"] = out["month"] + pd.DateOffset(months=1)
    out["target_change_1m_pct"] = np.where(
        out["price_ngn"] > 0,
        (out["target_price_ngn_1m"] - out["price_ngn"]) / out["price_ngn"],
        np.nan,
    )
    out["target_log_change_1m"] = np.where(
        (out["price_ngn"] > 0) & (out["target_price_ngn_1m"] > 0),
        np.log(out["target_price_ngn_1m"] / out["price_ngn"]),
        np.nan,
    )
    for lag in (1, 3, 12):
        col = f"price_lag_{lag}m"
        if col in out.columns:
            out[f"price_momentum_{lag}m_pct"] = np.where(
                out[col] > 0,
                (out["price_ngn"] - out[col]) / out[col],
                np.nan,
            )

    out["year"] = out["month"].dt.year.astype("int16")
    out["month_number"] = out["month"].dt.month.astype("int8")
    angle = 2.0 * math.pi * (out["month_number"].astype(float) - 1.0) / 12.0
    out["month_sin"] = np.sin(angle)
    out["month_cos"] = np.cos(angle)
    lag_columns = [f"price_lag_{lag}m" for lag in lags if f"price_lag_{lag}m" in out.columns]
    out["lag_feature_count"] = out[lag_columns].notna().sum(axis=1).astype("int8") if lag_columns else 0
    return out


def _market_month_counts(
    labels: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    require_next_target: bool,
) -> pd.DataFrame:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    work = labels[(labels["month"] >= start) & (labels["month"] < end)].copy()
    if require_next_target and "target_price_ngn_1m" in work.columns:
        work = work[work["target_price_ngn_1m"].notna()]
    return (
        work.groupby(["admin1", "admin2", "market", "market_id", "latitude", "longitude", "month"], as_index=False)
        .size()
        .rename(columns={"size": "label_rows"})
        .sort_values(["month", "label_rows", "market_id"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


def _spread_months(counts: pd.DataFrame, limit: int) -> pd.DataFrame:
    if counts.empty or limit <= 0:
        return counts.head(0).copy()
    months = sorted(pd.to_datetime(counts["month"].unique()))
    if len(months) <= limit:
        chosen_months = months
    else:
        indices = np.linspace(0, len(months) - 1, num=limit)
        chosen_months = [months[int(round(i))] for i in indices]
        chosen_months = list(dict.fromkeys(chosen_months))

    chosen_rows: list[pd.DataFrame] = []
    chosen_keys: set[tuple[int, pd.Timestamp]] = set()
    for month in chosen_months:
        group = counts[counts["month"].eq(month)].sort_values(
            ["label_rows", "market_id"], ascending=[False, True]
        )
        if not group.empty:
            row = group.head(1)
            chosen_rows.append(row)
            chosen_keys.add((int(row.iloc[0]["market_id"]), pd.Timestamp(month)))

    chosen = pd.concat(chosen_rows, ignore_index=True) if chosen_rows else counts.head(0).copy()
    if len(chosen) < limit:
        remaining = counts[
            ~counts.apply(lambda r: (int(r["market_id"]), pd.Timestamp(r["month"])) in chosen_keys, axis=1)
        ].copy()
        remaining["rank_in_month"] = remaining.groupby("month")["label_rows"].rank(
            method="first", ascending=False
        )
        remaining = remaining.sort_values(
            ["rank_in_month", "month", "label_rows", "market_id"],
            ascending=[True, True, False, True],
        )
        chosen = pd.concat([chosen, remaining.head(limit - len(chosen))], ignore_index=True)

    return chosen.sort_values(["month", "label_rows", "market_id"], ascending=[True, False, True]).head(limit).reset_index(drop=True)


def _spread_state_months(counts: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Greedy deterministic sampler balancing time, state and market reuse."""
    if counts.empty or limit <= 0:
        return counts.head(0).copy()
    work = counts.copy().reset_index(drop=True)
    months = sorted(pd.to_datetime(work["month"].unique()))
    selected: list[int] = []
    selected_set: set[int] = set()
    state_use: Counter[str] = Counter()
    market_use: Counter[int] = Counter()

    while len(selected) < limit:
        progress = False
        for month in months:
            idxs = work.index[work["month"].eq(month) & ~work.index.isin(selected_set)].tolist()
            if not idxs:
                continue
            ranked = sorted(
                idxs,
                key=lambda idx: (
                    state_use[str(work.at[idx, "admin1"])],
                    market_use[int(work.at[idx, "market_id"])],
                    -int(work.at[idx, "label_rows"]),
                    str(work.at[idx, "admin1"]),
                    int(work.at[idx, "market_id"]),
                ),
            )
            chosen = ranked[0]
            selected.append(chosen)
            selected_set.add(chosen)
            state_use[str(work.at[chosen, "admin1"])] += 1
            market_use[int(work.at[chosen, "market_id"])] += 1
            progress = True
            if len(selected) >= limit:
                break
        if not progress:
            break

    result = work.loc[selected].copy() if selected else work.head(0).copy()
    return result.sort_values(["month", "admin1", "label_rows", "market_id"], ascending=[True, True, False, True]).reset_index(drop=True)


def select_market_months(
    labels: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    limit: int | None = None,
    require_next_target: bool = True,
    spread_across_months: bool = False,
    spread_across_states: bool = False,
) -> pd.DataFrame:
    counts = _market_month_counts(
        labels,
        start_date=start_date,
        end_date=end_date,
        require_next_target=require_next_target,
    )
    if limit is not None and limit > 0:
        if spread_across_states:
            return _spread_state_months(counts, limit)
        if spread_across_months:
            return _spread_months(counts, limit)
        counts = counts.head(limit)
    return counts.reset_index(drop=True)
