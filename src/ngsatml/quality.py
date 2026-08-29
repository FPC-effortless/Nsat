from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

QUALITY_FACTOR_THRESHOLD = 5.0


def _national_split(values: pd.Series) -> pd.Series:
    years = pd.to_datetime(values).dt.year
    return pd.Series(
        np.select(
            [years <= 2020, years == 2021, years == 2022, years >= 2023],
            ["train", "validation", "test", "recent_ood"],
            default="other",
        ),
        index=values.index,
        dtype="object",
    )


def add_quality_columns(frame: pd.DataFrame, factor_threshold: float = QUALITY_FACTOR_THRESHOLD) -> pd.DataFrame:
    out = frame.copy()
    out["split_national"] = _national_split(out["month"])
    out["target_split_national"] = _national_split(out["target_month"])
    out["national_target_within_split"] = out["split_national"].eq(out["target_split_national"])

    price = pd.to_numeric(out["price_ngn"], errors="coerce")
    target = pd.to_numeric(out["target_price_ngn_1m"], errors="coerce")
    valid_target = (price > 0) & (target > 0)
    target_factor = pd.Series(np.nan, index=out.index, dtype=float)
    target_factor.loc[valid_target] = np.maximum(
        target.loc[valid_target] / price.loc[valid_target],
        price.loc[valid_target] / target.loc[valid_target],
    )
    out["quality_target_factor"] = target_factor

    lag = pd.to_numeric(out.get("price_lag_1m"), errors="coerce")
    valid_lag = (price > 0) & (lag > 0)
    input_factor = pd.Series(np.nan, index=out.index, dtype=float)
    input_factor.loc[valid_lag] = np.maximum(
        price.loc[valid_lag] / lag.loc[valid_lag],
        lag.loc[valid_lag] / price.loc[valid_lag],
    )
    out["quality_input_factor_1m"] = input_factor
    out["quality_extreme_target_change"] = out["quality_target_factor"].gt(float(factor_threshold))
    out["quality_extreme_input_jump_1m"] = out["quality_input_factor_1m"].gt(float(factor_threshold))
    lag_count = pd.to_numeric(out.get("lag_feature_count", 0), errors="coerce").fillna(0)
    out["quality_sparse_history"] = lag_count.lt(2)
    out["quality_suspicious"] = out["quality_extreme_target_change"] | out["quality_extreme_input_jump_1m"]
    return out


def _state_weights(train: pd.DataFrame) -> dict[str, float]:
    if train.empty:
        return {}
    counts = train["admin1"].value_counts()
    raw = len(train) / (len(counts) * counts)
    weights = np.sqrt(raw)
    mean_applied = train["admin1"].map(weights.to_dict()).mean()
    if mean_applied > 0:
        weights = weights / mean_applied
    weights = weights.clip(lower=0.5, upper=5.0)
    return {str(k): float(v) for k, v in weights.to_dict().items()}


def _write(frame: pd.DataFrame, base: Path) -> None:
    frame.to_csv(base.with_suffix(".csv"), index=False)
    frame.to_parquet(base.with_suffix(".parquet"), index=False)


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().to_dict().items()}


def _build_regime(frame: pd.DataFrame, prefix: str, final_dir: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    national = frame[
        frame["split_national"].isin(["train", "validation", "test"])
        & frame["national_target_within_split"]
    ].copy()
    train_clean = national[(national["split_national"] == "train") & ~national["quality_suspicious"]].copy()
    weights = _state_weights(train_clean)
    national["sample_weight_state"] = national["admin1"].map(weights).fillna(1.0).astype(float)
    train_clean = national[(national["split_national"] == "train") & ~national["quality_suspicious"]].copy()
    evaluation = national[national["split_national"].isin(["validation", "test"])].copy()
    recent_ood = frame[
        frame["split_national"].eq("recent_ood") & frame["national_target_within_split"]
    ].copy()

    _write(national, final_dir / f"{prefix}_national_ready")
    _write(train_clean, final_dir / f"{prefix}_national_train_clean")
    _write(evaluation, final_dir / f"{prefix}_national_eval")
    _write(recent_ood, final_dir / f"{prefix}_recent_ood")

    report = {
        "rows": int(len(national)),
        "train_clean_rows": int(len(train_clean)),
        "eval_rows": int(len(evaluation)),
        "split_rows": _counts(national["split_national"]) if len(national) else {},
        "states": int(national["admin1"].nunique()) if len(national) else 0,
        "markets": int(national["market_id"].nunique()) if len(national) else 0,
        "commodities": int(national["commodity"].nunique()) if len(national) else 0,
        "evaluation_states": int(evaluation["admin1"].nunique()) if len(evaluation) else 0,
        "evaluation_markets": int(evaluation["market_id"].nunique()) if len(evaluation) else 0,
        "top_state_share": float(national["admin1"].value_counts(normalize=True).max()) if len(national) else 0.0,
        "top_two_state_share": float(national["admin1"].value_counts(normalize=True).head(2).sum()) if len(national) else 0.0,
        "state_training_weights": weights,
        "weight_policy": "sqrt inverse-state-frequency, normalized then clipped to [0.5, 5.0]",
        "recent_ood_rows": int(len(recent_ood)),
        "recent_ood_states": int(recent_ood["admin1"].nunique()) if len(recent_ood) else 0,
    }
    return report, {
        "national": national,
        "train_clean": train_clean,
        "evaluation": evaluation,
        "recent_ood": recent_ood,
    }


def build_quality_views(final_dir: str | Path, factor_threshold: float = QUALITY_FACTOR_THRESHOLD) -> dict[str, Any]:
    final = Path(final_dir)
    core_path = final / "nsat_price_core_model_ready.parquet"
    satellite_path = final / "nsat_market_satellite_usable.parquet"
    if not core_path.exists() or not satellite_path.exists():
        raise FileNotFoundError("Run the Nsat dataset build before quality post-processing")

    core = add_quality_columns(pd.read_parquet(core_path), factor_threshold)
    satellite = add_quality_columns(pd.read_parquet(satellite_path), factor_threshold)
    _write(core, final / "nsat_price_core_model_ready_qc")
    _write(satellite, final / "nsat_market_satellite_usable_qc")

    core_report, core_views = _build_regime(core, "nsat_price_core", final)
    sat_report, sat_views = _build_regime(satellite, "nsat_satellite", final)

    report = {
        "version": "1.1-qc",
        "outlier_policy": {
            "factor_threshold": float(factor_threshold),
            "definition": "quality_suspicious marks >threshold adjacent-month target or 1m-input price ratios after canonical unit normalization",
            "training_policy": "exclude suspicious rows only from default training; do not target-filter validation/test or recent OOD",
        },
        "full_core": {
            "rows": int(len(core)),
            "quality_suspicious_rows": int(core["quality_suspicious"].sum()),
            "quality_suspicious_rate": float(core["quality_suspicious"].mean()) if len(core) else 0.0,
            "extreme_target_rows": int(core["quality_extreme_target_change"].sum()),
            "extreme_input_jump_rows": int(core["quality_extreme_input_jump_1m"].sum()),
        },
        "national_regime": core_report,
        "satellite_national_regime": sat_report,
        "recent_ood": {
            "definition": "2023-2025 rows are a separate recent temporal/geographic OOD regime because WFP source coverage shifts strongly toward northeastern Nigeria",
            "price_rows": int(len(core_views["recent_ood"])),
            "price_states": int(core_views["recent_ood"]["admin1"].nunique()) if len(core_views["recent_ood"]) else 0,
            "satellite_rows": int(len(sat_views["recent_ood"])),
            "satellite_states": int(sat_views["recent_ood"]["admin1"].nunique()) if len(sat_views["recent_ood"]) else 0,
        },
        "known_limitations": [
            "WFP covers 14 Nigerian states, not all 36 states plus FCT.",
            "The raw price core is concentrated in Borno and Yobe; use sample_weight_state and per-state reporting for broad-national training.",
            "From 2023 onward WFP coverage is primarily Adamawa, Borno and Yobe, so recent data are OOD rather than a nationally representative test.",
            "Sentinel features are local land-surface context around markets, not field-level crop-yield labels.",
            "Large price jumps remain in audit/evaluation outputs because some may be real shocks; only default training excludes >threshold transitions.",
        ],
    }
    (final / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    manifest_path = final / "modeling_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.update({
        "recommended_national_files": {
            "train": "nsat_price_core_national_train_clean.parquet",
            "validation_test": "nsat_price_core_national_eval.parquet",
            "recent_ood": "nsat_price_core_recent_ood.parquet",
        },
        "recommended_satellite_files": {
            "train": "nsat_satellite_national_train_clean.parquet",
            "validation_test": "nsat_satellite_national_eval.parquet",
            "recent_ood": "nsat_satellite_recent_ood.parquet",
        },
        "national_split_policy": "train=2015-2020, validation=2021, test=2022; 2023-2025 is recent OOD",
        "training_weight": "sample_weight_state uses sqrt inverse state frequency capped at 5x",
        "quality_policy": "default training excludes quality_suspicious; evaluation remains unfiltered by observed target",
    })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Nsat quality-controlled training and evaluation views")
    parser.add_argument("--input-dir", default="data/dataset/final")
    parser.add_argument("--factor-threshold", type=float, default=QUALITY_FACTOR_THRESHOLD)
    args = parser.parse_args()
    print(json.dumps(build_quality_views(args.input_dir, args.factor_threshold), indent=2))


if __name__ == "__main__":
    main()
