# Nsat model plan

## Primary research question

Does open Nigerian state context, climate, and satellite information add genuinely out-of-time predictive value beyond price history and seasonality?

Nsat now supports two related but statistically distinct task families.

## Task A — market/commodity price forecasting

For market `m`, commodity `c`, month `t`:

`y[m,t,c] = log(price[m,t+1,c] / price[m,t,c])`

Primary supervision: WFP market-price data.

### Baseline A0 — seasonal naive

- previous-year / last-observed seasonal rule where exact calendar history exists.

### Baseline A1 — price history

- 1/2/3/6/12-month exact-calendar lags
- 1/3/12-month momentum
- calendar month
- state/market/commodity/unit/price-type identity

### Model A2 — + nationwide NBS state context

Attach only predictor-month or historical state-level values to each WFP market row:

- Food CPI current/lagged values within the same CPI regime
- CoHD current/lagged values

Never attach `target_food_cpi_*` or `target_cohd_*` as predictors. The repeated state feature is contextual information; the direct supervised target remains the market/commodity WFP target.

### Model A3 — + climate

- NASA POWER precipitation and temperature features
- rainfall/climate lags/anomalies where available

### Model A4 — + satellite

- Sentinel-2 NDVI/NDWI/NBR and band summaries
- radar/terrain/cropland features as added
- cloud/valid-pixel diagnostics

Satellite signal is justified only if A4 improves genuinely out-of-time results over A1–A3.

## Task B — state Food CPI forecasting

Grain: one state-month row.

Recommended target:

`log(FoodCPI[s,t+1] / FoodCPI[s,t])`

Raw CPI levels must never be chained across the 2025 rebasing boundary. Train/evaluate within `index_regime` or on regime-safe relative changes.

Predictors should be aggregated to state-month before joining the NBS target. Do not duplicate one state label across many patches/markets as independent supervised rows.

## Task C — state CoHD forecasting

Grain: one state-month row.

Recommended target:

`log(CoHD[s,t+1] / CoHD[s,t])`

Compare against simple lag/seasonality baselines before adding climate or Earth observation features.

## Algorithms

The state-level panels are modest. Start with classical/small models:

1. seasonal naive
2. ridge/elastic net
3. gradient-boosted trees / CatBoost
4. only after robust evidence of added signal: temporal convolution, small transformer, or multimodal encoder

## Evaluation

Use contiguous temporal splits and rolling-origin checks where sample size permits. Report:

- MAE on log change
- directional accuracy
- price-level MAE/MAPE where meaningful
- performance by state and commodity
- calibration for material price spikes
- ablations: price history → +NBS state context → +climate → +satellite

## Leakage rules

- exact calendar joins only;
- no future-month NBS target columns as predictors;
- no row-shift lags across missing months;
- no random train/test row splits;
- no state-target pseudo-replication across market/patch observations;
- no raw CPI index comparisons across different base regimes.

## Falsification criterion

If climate/satellite features do not improve an out-of-time benchmark over price history plus NBS state context after reasonable feature engineering, do not claim that satellite information improves Nigerian food-price forecasting. The Earth-observation stack can still support crop-condition, flood, land-cover, and representation-learning tasks.
