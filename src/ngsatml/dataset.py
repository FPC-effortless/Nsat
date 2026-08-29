from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .power import DEFAULT_PARAMETERS, POWER_MONTHLY_URL, load_power_for_markets
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


def _attach_splits(frame: pd.DataFrame, split_cfg: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    out["split"] = _assign_split(out["month"], split_cfg)
    if "target_month" in out.columns:
        out["target_split"] = _assign_split(out["target_month"], split_cfg)
        out["target_within_split"] = out["split"].eq(out["target_split"])
    else:
        out["target_split"] = out["split"]
        out["target_within_split"] = True
    return out


def _month_after(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return (ts.to_period("M") + 1).to_timestamp()


def _select_market_months_for_build(
    labels: pd.DataFrame,
    cfg: dict[str, Any],
    dcfg: dict[str, Any],
    limit: int | None,
) -> pd.DataFrame:
    require_target = bool(dcfg.get("require_next_target", True))
    balanced = bool(dcfg.get("balanced_temporal_sampling", False))
    spatial = bool(dcfg.get("balanced_spatial_sampling", False))
    split_cfg = cfg.get("split", {})
    if not balanced or not limit or not split_cfg.get("train_end") or not split_cfg.get("validation_end"):
        return select_market_months(
            labels,
            start_date=cfg["start_date"],
            end_date=cfg["end_date"],
            limit=int(limit) if limit else None,
            require_next_target=require_target,
            spread_across_months=bool(dcfg.get("spread_across_months", False)),
            spread_across_states=spatial,
        )

    train_start = pd.Timestamp(cfg["start_date"])
    validation_start = _month_after(split_cfg["train_end"])
    test_start = _month_after(split_cfg["validation_end"])
    end = pd.Timestamp(cfg["end_date"])
    ranges = [
        ("train", train_start, validation_start),
        ("validation", validation_start, test_start),
        ("test", test_start, end),
    ]

    base = int(limit) // len(ranges)
    remainder = int(limit) % len(ranges)
    parts: list[pd.DataFrame] = []
    for idx, (name, start, stop) in enumerate(ranges):
        quota = base + (1 if idx < remainder else 0)
        if quota <= 0 or start >= stop:
            continue
        part = select_market_months(
            labels,
            start_date=start.isoformat(),
            end_date=stop.isoformat(),
            limit=quota,
            require_next_target=require_target,
            spread_across_months=True,
            spread_across_states=spatial,
        )
        if part.empty:
            raise RuntimeError(f"Balanced sampling produced no {name} market-months for {start.date()}..{stop.date()}")
        part = part.copy()
        part["selection_split"] = name
        parts.append(part)

    if not parts:
        return pd.DataFrame()
    selected = pd.concat(parts, ignore_index=True)
    return selected.sort_values(["month", "admin1", "label_rows", "market_id"], ascending=[True, True, False, True]).reset_index(drop=True)


def _write_frame(frame: pd.DataFrame, csv_path: Path, parquet_path: Path) -> None:
    csv_frame = frame.copy()
    for col in ["month", "target_month"]:
        if col in csv_frame.columns:
            csv_frame[col] = pd.to_datetime(csv_frame[col]).dt.strftime("%Y-%m-%d")
    csv_frame.to_csv(csv_path, index=False)
    frame.to_parquet(parquet_path, index=False)


def _data_dictionary() -> dict[str, Any]:
    return {
        "grain": "one market x month x commodity x canonical unit x price type",
        "canonical_units": {
            "kg": "Naira per kilogram; package units such as 500 G, 2.5 KG, 50 KG and 100 KG are normalized before aggregation",
            "l": "Naira per litre",
            "item": "Naira per item; package counts such as 30 pcs and 100 Tubers are normalized per item",
        },
        "targets": {
            "target_price_ngn_1m": "exact next-calendar-month price in the same canonical unit",
            "target_change_1m_pct": "fractional next-month price change",
            "target_log_change_1m": "log(next-month price/current price); recommended regression target when comparing commodities",
        },
        "price_history": "price_lag_*m are exact calendar lags and never bridge missing months",
        "target_within_split": "true only when both predictor month and target month belong to the same temporal split",
        "power": {
            "source": POWER_MONTHLY_URL,
            "parameters": list(DEFAULT_PARAMETERS),
            "note": "NASA POWER monthly point data at each WFP market coordinate; POWER-provided native units are retained. power_precip_est_mm_month multiplies PRECTOTCORR by days in month.",
        },
        "sentinel2": "monthly market-coordinate patch statistics from Earth Search Sentinel-2 L2A COGs; vegetation indices are local context, not direct crop-field yield labels",
    }


def _modeling_manifest() -> dict[str, Any]:
    return {
        "recommended_primary_task": "predict target_log_change_1m",
        "recommended_baseline": "gradient-boosted trees or CatBoost on price-core before neural/multimodal models",
        "split_policy": "strict temporal split; rows whose 1-month target crosses a train/validation or validation/test boundary are excluded from model-ready outputs",
        "do_not_use_as_features": [
            "target_price_ngn_1m", "target_change_1m_pct", "target_log_change_1m", "target_month", "target_split"
        ],
        "high_value_feature_groups": {
            "identity": ["admin1", "admin2", "market_id", "commodity", "category", "unit", "pricetype"],
            "price_history": ["price_ngn", "price_lag_1m", "price_lag_2m", "price_lag_3m", "price_lag_6m", "price_lag_12m", "price_momentum_1m_pct", "price_momentum_3m_pct", "price_momentum_12m_pct"],
            "seasonality": ["year", "month_number", "month_sin", "month_cos"],
            "climate": ["power_prectotcorr", "power_precip_est_mm_month", "power_t2m", "power_t2m_max", "power_t2m_min", "power_rh2m", "power_ws10m", "power_allsky_sfc_sw_dwn"],
            "satellite": ["s2_ndvi_median", "s2_ndwi_median", "s2_nbr_median", "s2_red_median", "s2_green_median", "s2_nir_median", "s2_swir16_median", "s2_clear_pixel_fraction"],
        },
        "evaluation": ["MAE on target_log_change_1m", "direction accuracy", "MAE/MAPE on target_price_ngn_1m by commodity", "performance by state and commodity", "price-history-only ablation vs +climate vs +Sentinel"],
    }


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
    lags = tuple(int(x) for x in dcfg.get("price_lags_months", [1, 2, 3, 6, 12]))
    labels = add_calendar_targets(labels, lags)
    labels = _attach_splits(labels, cfg.get("split", {}))

    end = pd.Timestamp(cfg["end_date"])
    core_start = pd.Timestamp(dcfg.get("core_start_date", cfg["start_date"]))
    core = labels[(labels["month"] >= core_start) & (labels["month"] < end)].copy()
    core_model = core[core["target_price_ngn_1m"].notna() & core["target_within_split"]].copy()

    climate_report: dict[str, Any] = {
        "enabled": False,
        "markets_requested": 0,
        "markets_succeeded": 0,
        "markets_failed": 0,
        "source": POWER_MONTHLY_URL,
    }
    climate = pd.DataFrame(columns=["market_id", "month"])
    if bool(dcfg.get("include_power_climate", False)) and not core.empty:
        markets = core[["market_id", "latitude", "longitude"]].drop_duplicates("market_id")
        climate, climate_report = load_power_for_markets(
            markets,
            start_date=core_start,
            end_date=end,
            cache_dir=raw_dir / "power",
        )
        climate_report["enabled"] = True
        if not climate.empty:
            core = core.merge(climate, on=["market_id", "month"], how="left")
            core_model = core_model.merge(climate, on=["market_id", "month"], how="left")

    power_cols = [c for c in core_model.columns if c.startswith("power_")]
    if power_cols:
        core_model["power_climate_available"] = core_model[power_cols].notna().any(axis=1)
        core["power_climate_available"] = core[power_cols].notna().any(axis=1)
    else:
        core_model["power_climate_available"] = False
        core["power_climate_available"] = False

    core_all_csv = final_dir / "nsat_price_core_all.csv"
    core_all_parquet = final_dir / "nsat_price_core_all.parquet"
    core_ready_csv = final_dir / "nsat_price_core_model_ready.csv"
    core_ready_parquet = final_dir / "nsat_price_core_model_ready.parquet"
    _write_frame(core, core_all_csv, core_all_parquet)
    _write_frame(core_model, core_ready_csv, core_ready_parquet)
    if not climate.empty:
        _write_frame(climate, final_dir / "power_market_month_features.csv", final_dir / "power_market_month_features.parquet")

    selection_pool = labels[
        labels["target_price_ngn_1m"].notna() & labels["target_within_split"]
    ].copy()
    configured_limit = dcfg.get("market_month_limit")
    limit = market_month_limit if market_month_limit is not None else configured_limit
    selected = _select_market_months_for_build(selection_pool, cfg, dcfg, int(limit) if limit else None)
    if selected.empty:
        start = pd.Timestamp(cfg["start_date"])
        period = selection_pool[(selection_pool["month"] >= start) & (selection_pool["month"] < end)]
        diagnostics = {
            "normalized_rows": int(len(labels)),
            "period_rows": int(len(period)),
            "period_months": sorted(period["month"].dt.strftime("%Y-%m").unique().tolist())[-12:] if len(period) else [],
            "states": sorted(period["admin1"].unique().tolist()) if len(period) else [],
            "units": sorted(period["unit"].dropna().astype(str).unique().tolist()) if len(period) else [],
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
        if hasattr(row, "selection_split"):
            base["selection_split"] = row.selection_split
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
            f"{getattr(row, 'selection_split', 'sample')} | {row.admin1} | {row.market} | "
            f"{pd.Timestamp(row.month).strftime('%Y-%m')} | "
            f"S2={features.get('s2_status')} candidates={features.get('s2_candidate_scenes', 0)}"
        )

    satellite = pd.DataFrame(feature_rows)
    selected_keys = selected[["market_id", "month"]].drop_duplicates()
    enriched = core_model.merge(selected_keys, on=["market_id", "month"], how="inner")
    enriched = enriched.merge(
        satellite.drop(columns=["admin1", "admin2", "market", "latitude", "longitude", "selection_split"], errors="ignore"),
        on=["market_id", "month"],
        how="left",
    )

    ordered_front = [
        "split", "month", "target_month", "admin1", "admin2", "market", "market_id", "latitude", "longitude",
        "category", "commodity", "commodity_id", "unit", "unit_family", "pricetype", "currency", "price_ngn", "target_price_ngn_1m",
        "target_change_1m_pct", "target_log_change_1m", "price_lag_1m", "price_lag_2m", "price_lag_3m", "price_lag_6m", "price_lag_12m",
        "price_momentum_1m_pct", "price_momentum_3m_pct", "price_momentum_12m_pct", "lag_feature_count", "price_observations",
        "s2_status", "s2_scene_id", "s2_scene_datetime", "s2_scene_cloud_cover", "s2_clear_pixel_fraction",
    ]
    front = [c for c in ordered_front if c in enriched.columns]
    rest = [c for c in enriched.columns if c not in front]
    enriched = enriched[front + rest]

    all_csv = final_dir / "nsat_market_satellite_all.csv"
    all_parquet = final_dir / "nsat_market_satellite_all.parquet"
    usable_csv = final_dir / "nsat_market_satellite_usable.csv"
    usable_parquet = final_dir / "nsat_market_satellite_usable.parquet"
    _write_frame(enriched, all_csv, all_parquet)
    usable = enriched[enriched["s2_status"].eq("ok")].copy()
    _write_frame(usable, usable_csv, usable_parquet)

    satellite.to_csv(final_dir / "satellite_market_month_features.csv", index=False)
    selected.to_csv(final_dir / "selected_market_months.csv", index=False)
    (final_dir / "data_dictionary.json").write_text(json.dumps(_data_dictionary(), indent=2), encoding="utf-8")
    (final_dir / "modeling_manifest.json").write_text(json.dumps(_modeling_manifest(), indent=2), encoding="utf-8")

    split_rows = {str(k): int(v) for k, v in usable["split"].value_counts().to_dict().items()}
    split_market_months = (
        usable[["split", "market_id", "month"]].drop_duplicates()["split"].value_counts().to_dict()
        if not usable.empty else {}
    )
    core_split_rows = {str(k): int(v) for k, v in core_model["split"].value_counts().to_dict().items()}
    core_state_rows = {str(k): int(v) for k, v in core_model["admin1"].value_counts().sort_index().to_dict().items()}
    selected_state_market_months = {
        str(k): int(v) for k, v in selected["admin1"].value_counts().sort_index().to_dict().items()
    }
    climate_ready = int(core_model["power_climate_available"].sum()) if "power_climate_available" in core_model else 0

    summary = {
        "version": "1.0.0",
        "wfp_source": WFP_CSV_URL,
        "sentinel_stac": "https://earth-search.aws.element84.com/v1",
        "power_source": POWER_MONTHLY_URL,
        "raw_wfp_rows": int(len(raw)),
        "normalized_rows_total": int(len(labels)),
        "core_start_date": str(core_start.date()),
        "end_date": cfg["end_date"],
        "core_rows_all": int(len(core)),
        "core_rows_model_ready": int(len(core_model)),
        "core_split_rows": core_split_rows,
        "core_states": int(core_model["admin1"].nunique()),
        "core_markets": int(core_model["market_id"].nunique()),
        "core_commodities": int(core_model["commodity"].nunique()),
        "core_state_rows": core_state_rows,
        "core_duplicate_label_keys": int(core_model.duplicated(["market_id", "month", "commodity", "unit", "pricetype"]).sum()),
        "power_climate": climate_report,
        "core_rows_with_power": climate_ready,
        "core_power_row_coverage": float(climate_ready / len(core_model)) if len(core_model) else 0.0,
        "balanced_temporal_sampling": bool(dcfg.get("balanced_temporal_sampling", False)),
        "balanced_spatial_sampling": bool(dcfg.get("balanced_spatial_sampling", False)),
        "selected_market_months": int(len(selected)),
        "selected_state_market_months": selected_state_market_months,
        "enriched_rows": int(len(enriched)),
        "usable_rows": int(len(usable)),
        "split_rows": split_rows,
        "split_market_months": {str(k): int(v) for k, v in split_market_months.items()},
        "months": sorted(pd.to_datetime(usable["month"]).dt.strftime("%Y-%m-%d").unique().tolist()) if len(usable) else [],
        "s2_market_months_ok": int(satellite["s2_status"].eq("ok").sum()),
        "s2_market_months_total": int(len(satellite)),
        "states": sorted(usable["admin1"].dropna().astype(str).unique().tolist()),
        "markets": int(usable["market_id"].nunique()) if len(usable) else 0,
        "commodities": sorted(usable["commodity"].dropna().astype(str).unique().tolist()),
        "units": sorted(usable["unit"].dropna().astype(str).unique().tolist()),
        "pricetypes": sorted(usable["pricetype"].dropna().astype(str).unique().tolist()),
        "outputs": {
            "core_all_csv": str(core_all_csv),
            "core_all_parquet": str(core_all_parquet),
            "core_model_ready_csv": str(core_ready_csv),
            "core_model_ready_parquet": str(core_ready_parquet),
            "enriched_all_csv": str(all_csv),
            "enriched_all_parquet": str(all_parquet),
            "enriched_usable_csv": str(usable_csv),
            "enriched_usable_parquet": str(usable_parquet),
        },
    }
    (final_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
