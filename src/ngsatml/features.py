from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a.astype("float32") - b.astype("float32")) / (a.astype("float32") + b.astype("float32") + EPS)


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    return normalized_difference(nir, red)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return normalized_difference(green, nir)


def nbr(nir: np.ndarray, swir22: np.ndarray) -> np.ndarray:
    return normalized_difference(nir, swir22)


def aggregate_state_month(patches: pd.DataFrame) -> pd.DataFrame:
    required = {"state", "date"}
    missing = required - set(patches.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    work = patches.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["month"] = work["date"].dt.to_period("M").astype(str)
    numeric = [c for c in work.select_dtypes(include="number").columns if c not in {"centroid_lat", "centroid_lon"}]
    if not numeric:
        return work[["state", "month"]].drop_duplicates().sort_values(["state", "month"]).reset_index(drop=True)
    grouped = work.groupby(["state", "month"], as_index=False)[numeric].agg(["median", "mean", "std"])
    grouped.columns = ["_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col) for col in grouped.columns]
    grouped = grouped.rename(columns={"state_": "state", "month_": "month"})
    return grouped.reset_index(drop=True)
