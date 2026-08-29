# First model plan

## Research question

Does open satellite/environmental information add out-of-time predictive value for Nigerian state-level food-price movement beyond price history and seasonality?

## Target

For commodity c, state s, month t:

`y[s,t,c] = log(price[s,t,c]) - log(price[s,t-1,c])`

## Feature families

### Baseline A — price only

- previous 1, 2, 3, 6, 12 month prices/returns
- calendar month
- state ID
- commodity ID

### Baseline B — rainfall + price

- current and lagged IMERG monthly rainfall
- 30/60/90-day rainfall anomalies
- price features above

### Model C — satellite + rainfall + price

- Sentinel-2 NDVI/EVI/NDWI state-cropland summaries
- Sentinel-1 VV/VH summaries and change features
- cropland coverage fraction
- cloud/valid-pixel diagnostics
- elevation summaries
- rainfall features
- price features

## Algorithms

Do not start with a neural network. Dataset size at state-month level is modest.

1. seasonal naive
2. linear/ridge
3. gradient boosting
4. only after evidence of signal: temporal convolution/small transformer

## Evaluation

Use MAE on monthly log-price change, directional accuracy, RMSE, and accuracy/calibration for material price spikes. Use contiguous temporal validation and test periods and report per-state/per-commodity errors.

## Falsification criterion

If satellite features do not improve the out-of-time test set over lagged prices + rainfall after reasonable feature engineering, do not treat food-price forecasting as a validated satellite opportunity. The geospatial data can still support land-cover, flood, crop-condition, and representation-learning tasks.
