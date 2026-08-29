# Dataset Card: Nsat Nigeria Food-Price + Earth Observation Stack

## Summary

Nsat is a reproducible Nigeria-focused dataset pipeline for combining market food prices, nationwide state economic targets, climate, and Earth-observation features while preserving source grain and provenance.

The repository does not redistribute nationwide raw satellite archives.

## Supervised tasks

### Market-level task

Predict next-month WFP market/commodity price movement.

Grain:

`market × month × commodity × canonical unit × price type`

Targets include exact-next-calendar-month Naira price, fractional price change, and log price change.

### State-level tasks

Predict/nowcast NBS state-level Food CPI or Cost of a Healthy Diet.

Grain:

`state × month`

These are separate target families. Image patches and market observations must be aggregated to state-month before NBS targets are used as direct supervision.

## Geographic coverage

- WFP layer: whatever Nigerian states/markets are present after configured filters.
- NBS Food CPI: all 36 states + FCT.
- NBS CoHD: all 36 states + FCT.
- Satellite pilot/configuration can select subsets for compute control.

Validated NBS v2 build:

- Food CPI: October 2024–June 2026, 21 consecutive months, 777 rows.
- CoHD: October 2024–April 2026, 19 consecutive months, 703 rows.

Both contain exactly 37 states per materialized month and passed strict continuity/key/value checks.

## Inputs

- WFP Nigeria market food prices
- NBS State Food CPI
- NBS State Cost of a Healthy Diet
- Sentinel-2 L2A optical imagery
- Sentinel-1 GRD radar infrastructure
- NASA POWER monthly climate
- Copernicus DEM
- ESA WorldCover 2021
- NASA GPM IMERG extension path
- geoBoundaries Nigerian ADM1 polygons

NBS Selected Food Price Watch is retained only as national/zonal commodity validation because its published tables are not a full all-state commodity label matrix.

## NBS transport/provenance

Canonical source: National Bureau of Statistics, Nigeria.

Default hosted build transport: provenance-preserving Electric Sheep Africa Parquet mirrors on Hugging Face when NBS NADA is unreachable from GitHub runners. Original NBS resource IDs, URLs, source resource names, source sheets, and retrieval timestamps are retained in the row/source index.

## CPI regime handling

NBS rebased CPI for the 2025 publication cycle. Nsat stores an explicit `index_regime` and never constructs raw-index lags or targets across that boundary.

## Unit of supervision

- WFP: market-month-commodity.
- NBS Food CPI: state-month.
- NBS CoHD: state-month.
- Satellite patch: auxiliary observation, not a direct state-level label.

A state-month NBS label must not be copied onto many market/patch rows and counted as independent supervision.

## Intended uses

- Nigerian market food-price forecasting
- state food-inflation nowcasting/forecasting
- state healthy-diet cost modeling
- ablation studies for price history vs climate vs satellite signal
- agricultural-condition feature research
- geospatial representation learning
- later flood/land-cover extensions

## Out-of-scope claims

This dataset does not by itself establish:

- crop yield;
- causal agricultural effects;
- household-level food insecurity;
- individual farm outcomes;
- causal links between satellite features and price movements.

## Quality gates

The nationwide NBS builder checks:

- 37 canonical states per month;
- no internal month gaps within each CPI regime / CoHD series;
- no unknown states;
- no duplicate state-month keys;
- no null/non-positive target values;
- source provenance/index output;
- artifact contract and SHA256 checksums in GitHub Actions.

## Evaluation

Use contiguous temporal splits. Establish price-only/seasonal baselines first, then add state context, climate, and satellite features. Satellite value is supported only by genuinely out-of-time improvement over simpler baselines.
