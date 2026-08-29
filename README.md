# Nsat

Nigeria-focused satellite ML research and dataset engineering.

Nsat is building a reproducible pipeline that combines Nigerian satellite observations with weather, terrain, land-cover, and public socioeconomic/agricultural data for small-model experiments.

## First benchmark

The first supervised task is **state-month food-price risk**.

Satellite observations are sampled at patch level, aggregated to state-month features, then joined to National Bureau of Statistics (NBS) food-price targets. This avoids treating many image patches carrying the same state-level price as independent training labels.

### MVP geography

- Kano
- Kaduna
- Niger
- Benue
- Oyo

### Primary data sources

- Sentinel-2 L2A — optical multispectral imagery via Element 84 Earth Search STAC.
- Sentinel-1 GRD — radar imagery via Earth Search STAC.
- Copernicus DEM GLO-30 — elevation.
- ESA WorldCover 2021 — 10 m land-cover map for cropland candidate selection.
- geoBoundaries gbOpen NGA ADM1 — Nigerian state boundaries, CC BY 4.0.
- NASA GPM IMERG V07 — rainfall features.
- Nigeria NBS Selected Food Price Watch — state-level monthly price targets.

See `DATA_SOURCES.md` for source and licensing notes.

## What the current MVP does

The checked-in pipeline can:

1. validate source configuration;
2. download and filter Nigerian ADM1 boundaries;
3. query Sentinel-1 and Sentinel-2 scene metadata without downloading whole scenes;
4. save compact STAC manifests;
5. discover NBS downloadable resources;
6. compute core feature/aggregation utilities;
7. enforce temporal train/validation/test splitting;
8. run unit tests in GitHub Actions.

The current workflow does **not yet claim to materialize a complete satellite training table**. COG window extraction, WorldCover-constrained patch sampling, IMERG materialization, robust NBS spreadsheet parsing, and final benchmark training are the next implementation layers.

## Dataset design

### Patch table — auxiliary

One record per sampled observation patch:

- `state`
- `patch_id`
- `date`
- centroid coordinates
- Sentinel scene IDs and metadata
- source asset references
- derived optical/radar statistics when enabled

The patch table is not the direct supervised target table.

### State-month table — supervised

One record per state and month, containing aggregated vegetation/radar/weather features plus the corresponding NBS target.

## Install

Python 3.11+:

```bash
pip install -e '.[dev]'
```

For raster/geospatial processing later:

```bash
pip install -e '.[geo,dev]'
```

## Local commands

```bash
python -m ngsatml.cli sources
python -m ngsatml.cli boundaries --config configs/smoke.yaml
python -m ngsatml.cli catalog --config configs/smoke.yaml
python -m ngsatml.cli nbs --config configs/smoke.yaml
pytest -q
```

## GitHub Actions

- **CI** runs the unit test suite on pushes and pull requests.
- **Build source catalog** is manually runnable from the Actions tab. `smoke` mode covers Kano for a short period; `pilot` mode covers the five-state 2022–2026 pilot. The workflow uploads generated boundaries, STAC manifests, and the NBS resource index as an Actions artifact.

## Scientific guardrails

- Split forecasting data by time, not random rows.
- Keep state-level NBS labels at state-month granularity.
- Track missing satellite/cloud coverage explicitly.
- Never leak future-month prices into earlier features.
- Establish strong price-only and weather-only baselines before claiming satellite signal adds predictive value.

## Minimum benchmark

1. Seasonal naive baseline.
2. Lagged-price baseline.
3. Weather + lagged-price model.
4. Satellite + weather + lagged-price model.

Satellite inputs are justified only if model 4 improves genuinely out-of-time performance over the simpler baselines.
