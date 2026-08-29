from __future__ import annotations

import re

import pandas as pd

from .nbs import normalize_state

_SAFE_PATTERNS = (
    re.compile(r"^food_cpi$"),
    re.compile(r"^index_regime$"),
    re.compile(r"^food_cpi_lag_\d+m$"),
    re.compile(r"^cohd_ngn_person_day$"),
    re.compile(r"^cohd_lag_\d+m$"),
)


def _safe_context_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if any(pattern.match(c) for pattern in _SAFE_PATTERNS)]


def attach_nbs_state_context(
    market_rows: pd.DataFrame,
    state_month_targets: pd.DataFrame,
    *,
    market_state_col: str = "admin1",
) -> pd.DataFrame:
    """Attach leakage-safe NBS state-month context to market-level rows.

    The market/commodity row remains the unit of supervision. NBS state values
    are repeated only as shared contextual predictors. Future NBS target columns
    are intentionally not eligible for the join. `validate='many_to_one'`
    prevents silently multiplying market rows when the state target table is not
    unique at state-month grain.
    """
    required_market = {market_state_col, "month"}
    required_state = {"state", "month"}
    missing_market = required_market - set(market_rows.columns)
    missing_state = required_state - set(state_month_targets.columns)
    if missing_market:
        raise ValueError(f"Market frame missing required columns: {sorted(missing_market)}")
    if missing_state:
        raise ValueError(f"State target frame missing required columns: {sorted(missing_state)}")

    market = market_rows.copy()
    context = state_month_targets.copy()
    market["month"] = pd.to_datetime(market["month"]).dt.to_period("M").dt.to_timestamp()
    context["month"] = pd.to_datetime(context["month"]).dt.to_period("M").dt.to_timestamp()
    market["_nsat_state"] = market[market_state_col].map(normalize_state)
    context["_nsat_state"] = context["state"].map(normalize_state)

    safe = _safe_context_columns(list(context.columns))
    context = context[["_nsat_state", "month"] + safe].dropna(subset=["_nsat_state"])
    if context.duplicated(["_nsat_state", "month"]).any():
        dupes = context.loc[context.duplicated(["_nsat_state", "month"], keep=False), ["_nsat_state", "month"]]
        sample = dupes.drop_duplicates().head(10).to_dict(orient="records")
        raise ValueError(f"NBS state context is not unique at state-month grain: {sample}")

    merged = market.merge(context, on=["_nsat_state", "month"], how="left", validate="many_to_one")
    merged["nbs_state_context_available"] = merged[safe].notna().any(axis=1) if safe else False
    return merged.drop(columns=["_nsat_state"])
