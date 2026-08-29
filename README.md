# Nsat

Nigeria-focused satellite, environmental, and food-price ML dataset engineering.

Nsat builds reproducible training tables that combine Nigerian market prices, nationwide NBS economic targets, weather, and Earth-observation features without mirroring large raw satellite archives into Git.

## Current data products

### 1. Market food-price forecasting layer

Grain: **market × month × commodity × canonical unit × price type**.

Primary source: WFP Nigeria market-price history. The builder normalizes package units, creates exact-calendar price lags, next-month targets, temporal split metadata, NASA POWER climate features, and optional Sentinel-2 market-patch features.

### 2. Nationwide NBS state target layer

Grain: **state × month** for all **36 states + FCT**.

Two official NBS series are materialized independently:

- **Food CPI** — state-level food-price pressure. CPI index regimes are kept separate across the 2025 rebasing; raw index levels are never bridged blindly across the base change.
- **Cost of a Healthy Diet (CoHD)** — state-level Naira/person/day affordability-cost signal.

The validated v2 build currently materializes:

- Food CPI: 777 rows = 21 consecutive months × 37 states, October 2024 through June 2026.
- CoHD: 703 rows = 19 consecutive months × 37 states, October 2024 through April 2026.

The strict quality gate found no incomplete months, internal calendar gaps, unknown states, duplicate keys, null target values, or non-positive target values in that build.

### 3. Satellite/environmental enrichment

- Sentinel-2 L2A optical observations via Element 84 Earth Search STAC.
- Sentinel-1 GRD radar metadata/extraction infrastructure.
- NASA POWER climate features.
- Copernicus DEM and ESA WorldCover source registry/support.
- geoBoundaries Nigerian ADM1 boundaries.

Satellite imagery is streamed/cropped from remote assets rather than downloaded nationwide.

## Important correction: Selected Food Price Watch

NBS Selected Food Price Watch is **not** treated as a complete 37-state × commodity supervision matrix. Its downloadable report tables provide national/zonal item averages and selected state extrema, so Nsat uses it only for commodity-level validation where the published grain supports the comparison.

For nationwide state supervision, Nsat uses State Food CPI and CoHD instead. WFP remains the granular market/commodity target source.

## Provenance and transport resilience

Nigeria NBS remains the canonical publisher for CPI and CoHD. The NBS NADA host can be intermittently unreachable from GitHub-hosted runners, so the default reproducible build can use Electric Sheep Africa's Hugging Face Parquet copies as a **transport mirror**. Those mirrored rows retain the original NBS resource IDs, URLs, resource titles, retrieval timestamps, and source-sheet provenance.

Run with direct NBS refresh only when desired; source-network failures are recorded separately from data-quality failures.

## Install

Python 3.11+:

```bash
pip install -e '.[dev]'
```

## Core commands

```bash
python -m ngsatml.cli sources
python -m ngsatml.cli boundaries --config configs/smoke.yaml
python -m ngsatml.cli catalog --config configs/smoke.yaml
python -m ngsatml.cli dataset --config configs/dataset-smoke.yaml

# Build nationwide NBS target layer
python -m ngsatml.cli nbs-targets \
  --output data/nbs-targets \
  --start-date 2024-10-01 \
  --strict

pytest -q
```

## NBS target artifact contract

A successful build writes:

- `nbs_food_cpi.parquet` / `.csv`
- `nbs_cohd.parquet` / `.csv`
- `nbs_state_month_targets.parquet`
- `nbs_source_index.csv`
- `nbs_quality.json`
- `nbs_summary.json`
- workflow-generated `SHA256SUMS.txt`

Food CPI contains exact-calendar lags at 1/2/3/6/12 months and next-month index/change targets. CoHD uses the same temporal discipline. Missing months are never forward-filled.

## GitHub Actions

- **CI** — repository tests.
- **Build source catalog** — boundaries/STAC/source discovery.
- **Build training-grade dataset** — WFP + climate + optional Sentinel path.
- **Build nationwide NBS targets** — monthly scheduled/manual build with strict 37-state completeness, continuity, validity, artifact-contract, and checksum gates.

The NBS workflow publishes a compact validated artifact rather than raw upstream workbooks.

## Modeling hierarchy

For market-level forecasting, compare progressively:

1. seasonal/price-history baseline;
2. + nationwide NBS state context (Food CPI/CoHD, current or lagged only);
3. + weather;
4. + satellite features.

For state-level NBS forecasting, aggregate predictors to state-month before training. Do **not** replicate one state-month NBS label across many market or image-patch rows and treat those rows as independent labels.

## Scientific guardrails

- use contiguous temporal validation/test splits;
- use exact calendar lags rather than row shifts across missing months;
- never use next-month target columns as predictors;
- never compare raw Food CPI levels across index regimes as if the base were unchanged;
- preserve source provenance and missingness;
- treat NBS state targets, WFP market targets, and satellite patch observations as different grains;
- require out-of-time improvement over price-history and weather baselines before claiming satellite signal adds predictive value.
