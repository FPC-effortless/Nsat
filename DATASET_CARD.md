# Dataset Card: Nsat MVP

## Summary

A reproducible recipe for building Nigeria-focused Earth-observation features from open satellite/environmental sources and joining them to Nigerian state-level economic targets. The repository does not redistribute large raw satellite archives.

## Initial task

State-month food-price movement nowcasting/forecasting for selected Nigerian states.

## Geographic coverage

Initial pilot: Kano, Kaduna, Niger, Benue, and Oyo. Expansion to all 36 states + FCT follows validation.

## Inputs

- Sentinel-2 L2A optical imagery
- Sentinel-1 GRD SAR imagery
- Copernicus DEM
- ESA WorldCover 2021 cropland mask
- NASA GPM IMERG rainfall
- geoBoundaries ADM1 polygons

## Labels

Nigeria National Bureau of Statistics Selected Food Price Watch, normalized at state-month-commodity level when source tables can be parsed without ambiguous inference.

## Unit of supervision

State-month-commodity. Image patches are auxiliary observations and must be aggregated before joining state-level price labels.

## Intended uses

- satellite signal research for Nigerian food-price nowcasting
- agricultural-condition features
- geospatial representation learning
- later flood and land-cover extensions

## Out-of-scope claims

This dataset does not by itself establish crop yield, causal agricultural effects, household food insecurity, or individual farm outcomes.

## Evaluation

Use contiguous temporal splits. Compare against seasonal, lagged-price, and rainfall baselines before attributing value to satellite features.
