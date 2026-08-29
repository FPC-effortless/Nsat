from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .satellite import extract_market_month_s2
from .wfp import WFP_CSV_URL, add_calendar_targets, download_wfp_csv, normalize_prices, read_wfp_csv, select_market_months


def _dataset_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("dataset", {}) if isinstance(cfg.get("dataset", {}), dict) else {}


def _optional_filter(cfg: dict[str, Any], key: str, default: list[str] | None = None) -> list[str] | None:
    if key not in cfg:
        return default
    value = cfg[key]
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"dataset.{key} must be a list or null")
    return value or None


def _assign_split(month: pd.Series, split_cfg: dict[str, Any]) -> pd.Series:
    if not split_cfg:
        return pd.Series(["unspecified"] * len(month), index=month.index, dtype="object")
    train_end = pd.Timestamp(split_cfg.get("train_end")) if split_cfg.get("train_end") else None
    validation_end = pd.Timestamp(split_cfg.get("validation_end")) if split_cfg.get("validation_end") else None

    def one(value: pd.Timestamp) -> str:
        if train_end is not None and value <= train_end:
            return "train"
        if validation_end is not None and value <= validation_end:
            return "validation"
        return "test"

    return month.map(one)


def build_market_satellite_dataset(
    cfg: dict[str, Any],
    output_dir: str | Path,
    *,
    market_month_limit: int | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    raw_dir = out / "raw"
    final_dir = out / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    dcfg = _dataset_config(cfg)
    wfp_path = download_wfp_csv(raw_dir / "wfp_food_prices_nga.csv")
    raw = read_wfp_csv(wfp_path)
    labels = normalize_prices(
        raw,
        states=_optional_filter(dcfg, "states"),
        commodities=_optional_filter(dcfg, "commodities"),
        units=_optional_filter(dcfg, "units", ["KG"]),
        pricetypes=_optional_filter(dcfg, "pricetypes"),
    )
    labels = add_calendar_targets(labels, tuple(int(x) for x in dcfg.get("price_lags_months", [1, 3, 12])))

    configured_limit = dcfg.get("market_month_limit")
    limit = market_month_limit if market_month_limit is not None else configured_limit
    selected = select_market_months(
        labels,
        start_date=cfg["start_date"],
        end_date=cfg["end_date"],
        limit=int(limit) if limit else None,
        require_next_target=bool(dcfg.get("require_next_target", True)),
    )
    if selected.empty:
        start = pd.Timestamp(cfg["start_date"])
        end = pd.Timestamp(cfg["end_date"])
        period = labels[(labels["month"] >= start) & (labels["month"] < end)]
        diagnostics = {
            "normalized_rows": int(len(labels)),
            "period_rows": int(len(period)),
            "period_rows_with_next_month_target": int(period.get("target_price_ngn_1m", pd.Series(dtype=float)).notna().sum()),
            "period_months": sorted(period["month"].dt.strftime("%Y-%m").unique().tolist())[-12:] if len(period) else [],
            "units": sorted(period["unit"].dropna().astype(str).unique().tolist())[:25] if len(period) else [],
            "commodities": sorted(period["commodity"].dropna().astype(str).unique().tolist())[:50] if len(period) else [],
        }
        raise RuntimeError(f"No WFP market-months matched the configured period and filters: {json.dumps(diagnostics)}")

    s2_cfg = cfg.get("sentinel2", {})
    patch_radius_m = float(dcfg.get("patch_radius_m", 1280.0))
    patch_pixels = int(dcfg.get("patch_pixels", 64))
    max_scenes = int(dcfg.get("max_s2_scenes", 3))

    feature_rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        base = {
            "admin1": row.admin1,
            "admin2": row.admin2,
            "market": row.market,
            "market_id": int(row.market_id),
            "latitude": float(row.latitude),
            "longitude": float(row.longitude),
            "month": pd.Timestamp(row.month),
            "market_month_label_rows": int(row.label_rows),
        }
        features = extract_market_month_s2(
            float(row.longitude),
            float(row.latitude),
            row.month,
            collection=s2_cfg.get("collection", "sentinel-2-l2a"),
            cloud_cover_max=float(s2_cfg.get("cloud_cover_max", 70)),
            patch_radius_m=patch_radius_m,
            out_size=patch_pixels,
            max_scenes=max_scenes,
        )
        base.update(features)
        feature_rows.append(base)
        print(
            f"{row.admin1} | {row.market} | {pd.Timestamp(row.month).strftime('%Y-%m')} | "
            f"S2={features.get('s2_status')} candidates={features.get('s2_candidate_scenes', 0)}"
        )

    satellite = pd.DataFrame(feature_rows)
    selected_keys = selected[["market_id", "month"]].drop_duplicates()
    scoped_labels = labels.merge(selected_keys, on=["market_id", "month"], how="inner")
    dataset = scoped_labels.merge(
        satellite.drop(columns=["admin1", "admin2", "market", "latitude", "longitude"], errors="ignore"),
        on=["market_id", "month"],
        how="left",
    )
    dataset["split"] = _assign_split(dataset["month"], cfg.get("split", {}))
    dataset["month"] = pd.to_datetime(dataset["month"]).dt.strftime("%Y-%m-%d")

    ordered_front = [
        "split", "month", "admin1", "admin2", "market", "market_id", "latitude", "longitude",
        "commodity", "unit", "pricetype", "currency", "price_ngn", "target_price_ngn_1m",
        "target_change_1m_pct", "price_lag_1m", "price_lag_3m", "price_lag_12m", "price_observations",
        "s2_status", "s2_scene_id", "s2_scene_datetime", "s2_scene_cloud_cover", "s2_clear_pixel_fraction",
    ]
    front = [c for c in ordered_front if c in dataset.columns]
    rest = [c for c in dataset.columns if c not in front]
    dataset = dataset[front + rest]

    all_csv = final_dir / "nsat_market_satellite_all.csv"
    all_parquet = final_dir / "nsat_market_satellite_all.parquet"
    usable_csv = final_dir / "nsat_market_satellite_usable.csv"
    usable_parquet = final_dir / "nsat_market_satellite_usable.parquet"
    dataset.to_csv(all_csv, index=False)
    dataset.to_parquet(all_parquet, index=False)
    usable = dataset[dataset["s2_status"].eq("ok")].copy()
    usable.to_csv(usable_csv, index=False)
    usable.to_parquet(usable_parquet, index=False)

    summary = {
        "version": "0.2.0",
        "wfp_source": WFP_CSV_URL,
        "sentinel_stac": "https://earth-search.aws.element84.com/v1",
        "start_date": cfg["start_date"],
        "end_date": cfg["end_date"],
        "selected_market_months": int(len(selected)),
        "label_rows": int(len(dataset)),
        "usable_rows": int(len(usable)),
        "s2_market_months_ok": int(satellite["s2_status"].eq("ok").sum()),
        "s2_market_months_total": int(len(satellite)),
        "states": sorted(dataset["admin1"].dropna().astype(str).unique().tolist()),
        "markets": int(dataset["market_id"].nunique()),
        "commodities": sorted(dataset["commodity"].dropna().astype(str).unique().tolist()),
        "units": sorted(dataset["unit"].dropna().astype(str).unique().tolist()),
        "pricetypes": sorted(dataset["pricetype"].dropna().astype(str).unique().tolist()),
        "outputs": {
            "all_csv": str(all_csv),
            "all_parquet": str(all_parquet),
            "usable_csv": str(usable_csv),
            "usable_parquet": str(usable_parquet),
        },
    }
    (final_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    satellite.to_csv(final_dir / "satellite_market_month_features.csv", index=False)
    selected.to_csv(final_dir / "selected_market_months.csv", index=False)
    return summary
