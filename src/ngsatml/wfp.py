from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

WFP_CSV_URL = (
    "https://data.humdata.org/dataset/42db041f-7aaf-4ab4-961f-2a12096861e7/"
    "resource/12b51155-0cd3-4806-9924-61ede4077591/download/wfp_food_prices_nga.csv"
)

LABEL_KEYS = ["market_id", "commodity", "unit", "pricetype"]


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
    for col in ["latitude", "longitude", "price", "usdprice", "market_id"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    for col in ["admin1", "admin2", "market", "commodity", "unit", "pricetype", "currency"]:
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
    group_cols = [
        "admin1", "admin2", "market", "market_id", "latitude", "longitude",
        "month", "commodity", "unit", "pricetype", "currency",
    ]
    agg = {"price": ["median", "count"]}
    if "usdprice" in work.columns:
        agg["usdprice"] = ["median"]
    labels = work.groupby(group_cols, dropna=False).agg(agg).reset_index()
    labels.columns = [
        "_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col)
        for col in labels.columns
    ]
    labels = labels.rename(
        columns={
            "price_median": "price_ngn",
            "price_count": "price_observations",
            "usdprice_median": "price_usd",
        }
    )
    return labels.sort_values(["month", "admin1", "market", "commodity", "pricetype"]).reset_index(drop=True)


def add_calendar_targets(labels: pd.DataFrame, lags: tuple[int, ...] = (1, 3, 12)) -> pd.DataFrame:
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
    future["month"] = future["month"] - pd.DateOffset(months=1)
    future = future.rename(columns={"price_ngn": "target_price_ngn_1m"})
    out = out.merge(future, on=LABEL_KEYS + ["month"], how="left")
    out["target_change_1m_pct"] = np.where(
        out["price_ngn"] > 0,
        (out["target_price_ngn_1m"] - out["price_ngn"]) / out["price_ngn"],
        np.nan,
    )
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
        # Fill remaining quota by cycling chronologically through months, taking the
        # highest-coverage unused market within each month before taking a second one.
        remaining["rank_in_month"] = remaining.groupby("month")["label_rows"].rank(
            method="first", ascending=False
        )
        remaining = remaining.sort_values(
            ["rank_in_month", "month", "label_rows", "market_id"],
            ascending=[True, True, False, True],
        )
        chosen = pd.concat([chosen, remaining.head(limit - len(chosen))], ignore_index=True)

    return chosen.sort_values(["month", "label_rows", "market_id"], ascending=[True, False, True]).head(limit).reset_index(drop=True)


def select_market_months(
    labels: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    limit: int | None = None,
    require_next_target: bool = True,
    spread_across_months: bool = False,
) -> pd.DataFrame:
    counts = _market_month_counts(
        labels,
        start_date=start_date,
        end_date=end_date,
        require_next_target=require_next_target,
    )
    if limit is not None and limit > 0:
        if spread_across_months:
            return _spread_months(counts, limit)
        counts = counts.head(limit)
    return counts.reset_index(drop=True)
