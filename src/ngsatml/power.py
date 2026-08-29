from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

POWER_MONTHLY_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
DEFAULT_PARAMETERS = (
    "PRECTOTCORR",
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "WS10M",
    "ALLSKY_SFC_SW_DWN",
)


def _request_power(
    latitude: float,
    longitude: float,
    start_year: int,
    end_year: int,
    parameters: Iterable[str],
    session: requests.Session,
    retries: int = 4,
) -> dict:
    params = {
        "parameters": ",".join(parameters),
        "community": "AG",
        "longitude": f"{float(longitude):.6f}",
        "latitude": f"{float(latitude):.6f}",
        "format": "JSON",
        "start": str(int(start_year)),
        "end": str(int(end_year)),
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(POWER_MONTHLY_URL, params=params, timeout=90)
            if response.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            payload = response.json()
            if "properties" not in payload or "parameter" not in payload["properties"]:
                raise ValueError("NASA POWER response missing properties.parameter")
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"NASA POWER request failed after {retries} attempts: {last_error}")


def _parse_power_payload(payload: dict, market_id: int) -> pd.DataFrame:
    parameters = payload.get("properties", {}).get("parameter", {})
    keys: set[str] = set()
    for values in parameters.values():
        keys.update(str(k) for k in values.keys())

    rows: list[dict] = []
    for key in sorted(keys):
        if len(key) != 6 or not key.isdigit():
            continue
        year = int(key[:4])
        month_num = int(key[4:])
        if not 1 <= month_num <= 12:
            continue
        row: dict[str, object] = {
            "market_id": int(market_id),
            "month": pd.Timestamp(year=year, month=month_num, day=1),
        }
        for parameter, values in parameters.items():
            value = values.get(key)
            if value is None or float(value) <= -900:
                row[f"power_{parameter.lower()}"] = np.nan
            else:
                row[f"power_{parameter.lower()}"] = float(value)
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    precip_col = "power_prectotcorr"
    if precip_col in result.columns:
        result["power_precip_est_mm_month"] = result[precip_col] * result["month"].dt.days_in_month
    return result


def load_power_for_markets(
    markets: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    cache_dir: str | Path,
    parameters: Iterable[str] = DEFAULT_PARAMETERS,
    session: requests.Session | None = None,
) -> tuple[pd.DataFrame, dict]:
    required = {"market_id", "latitude", "longitude"}
    missing = required - set(markets.columns)
    if missing:
        raise ValueError(f"Market table missing columns for POWER enrichment: {sorted(missing)}")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    start_year = int(start.year)
    end_year = int((end - pd.Timedelta(days=1)).year)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    s = session or requests.Session()

    frames: list[pd.DataFrame] = []
    failures: list[dict] = []
    unique_markets = markets[["market_id", "latitude", "longitude"]].drop_duplicates("market_id").sort_values("market_id")
    for row in unique_markets.itertuples(index=False):
        target = cache / f"market_{int(row.market_id)}_{start_year}_{end_year}.json"
        try:
            if target.exists() and target.stat().st_size > 0:
                payload = json.loads(target.read_text(encoding="utf-8"))
            else:
                payload = _request_power(
                    float(row.latitude),
                    float(row.longitude),
                    start_year,
                    end_year,
                    parameters,
                    s,
                )
                target.write_text(json.dumps(payload), encoding="utf-8")
                time.sleep(0.05)
            frame = _parse_power_payload(payload, int(row.market_id))
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:  # Preserve partial data if one external point fails.
            failures.append({"market_id": int(row.market_id), "error": str(exc)})

    climate = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["market_id", "month"])
    if not climate.empty:
        climate = climate[(climate["month"] >= start) & (climate["month"] < end)].reset_index(drop=True)
    report = {
        "markets_requested": int(len(unique_markets)),
        "markets_succeeded": int(climate["market_id"].nunique()) if not climate.empty else 0,
        "markets_failed": int(len(failures)),
        "failures": failures[:20],
        "parameters": list(parameters),
        "source": POWER_MONTHLY_URL,
    }
    return climate, report
