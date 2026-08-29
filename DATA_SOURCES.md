# Data source registry

This file is operational guidance, not legal advice. Re-check upstream terms before commercial redistribution.

| Source | Role in Nsat | Access | Provenance / use note |
|---|---|---|---|
| WFP Nigeria food prices | Market × month × commodity supervision | HDX/WFP CSV | Keep original market/admin/commodity metadata; normalize package units before comparing prices |
| NBS Consumer Price Index | 36 states + FCT monthly Food CPI target/context layer | NBS NADA catalog 154 | Canonical publisher: National Bureau of Statistics, Nigeria; keep CPI base regimes separate |
| NBS Cost of a Healthy Diet | 36 states + FCT monthly affordability/cost target/context layer | NBS NADA catalog 146 | Canonical publisher: National Bureau of Statistics, Nigeria; Naira/person/day state average |
| Electric Sheep Africa NBS mirrors | Reliable Parquet transport for NBS CPI/CoHD | Hugging Face Dataset Server/Hub | Secondary transport only. Preserve original NBS resource IDs, URLs, sheet names and retrieval timestamps; cite original NBS source and mirror when redistributed |
| NBS Selected Food Price Watch | National/zonal commodity validation | NBS NADA catalog 162 / e-library | **Not assumed to be a complete 37-state × commodity matrix**; do not infer unpublished state rows |
| Copernicus Sentinel-2 L2A | Optical imagery | Earth Search / Copernicus Data Space STAC | Sentinel data are free, full and open; stream/crop assets instead of mirroring full scenes |
| Copernicus Sentinel-1 GRD | Radar imagery | Earth Search / AWS Open Data | Useful through cloud/night; preserve scene IDs and acquisition metadata |
| Copernicus DEM GLO-30 | Elevation | Earth Search | Verify Copernicus DEM attribution notice for downstream release |
| ESA WorldCover 2021 | Cropland/land-cover mask | AWS Open Data / Planetary Computer | CC BY 4.0; class 40 is cropland |
| geoBoundaries gbOpen NGA ADM1 | Nigerian state boundaries | geoBoundaries API | CC BY 4.0 |
| NASA POWER | Market-coordinate monthly climate | NASA POWER API | Retain parameter names/native units and source endpoint |
| NASA GPM IMERG V07 | Rainfall extension | NASA GES DISC/PPS | Generally open NASA mission data; free Earthdata/PPS account may be required for some access paths |

## Canonical endpoints

- WFP Nigeria food prices: `https://data.humdata.org/dataset/wfp-food-prices-for-nigeria`
- NBS CPI: `https://microdata.nigerianstat.gov.ng/index.php/catalog/154/related-materials`
- NBS CoHD: `https://microdata.nigerianstat.gov.ng/index.php/catalog/146/related-materials`
- NBS Selected Food Price Watch: `https://microdata.nigerianstat.gov.ng/index.php/catalog/162/related-materials`
- Electric Sheep Africa mirror namespace: `https://huggingface.co/electricsheepafrica`
- Earth Search STAC: `https://earth-search.aws.element84.com/v1`
- Copernicus Data Space STAC: `https://stac.dataspace.copernicus.eu/v1/`
- Planetary Computer STAC: `https://planetarycomputer.microsoft.com/api/stac/v1/`
- geoBoundaries NGA ADM1: `https://www.geoboundaries.org/api/current/gbOpen/NGA/ADM1/`
- NASA IMERG: `https://gpm.nasa.gov/data/directory`

## NBS target grain policy

Three Nsat price/economic sources have different grains and must remain distinguishable:

1. **WFP:** market × month × commodity × unit/price type.
2. **NBS Food CPI / CoHD:** state × month.
3. **Selected Food Price Watch:** published national/zonal commodity tables plus selected state extrema.

Do not turn state-level NBS targets into independent market- or patch-level labels. State features may be attached to market rows as shared context for a market-level WFP target, but direct NBS prediction must aggregate predictors to state-month.

## CPI rebasing policy

Nsat records an explicit `index_regime`. The pre-2025 NBS CPI and the rebased 2025+ CPI are not treated as one raw-index continuum. Lag and target construction occurs within regime. Relative/log changes are safe only when both observations belong to the same regime.

## Transport policy

The default GitHub target builder discovers and reads provenance-preserving NBS Parquet mirrors because the NBS NADA host can time out from hosted runners. This does **not** change the source of truth: NBS remains canonical. The source index records mirror repository plus original NBS resource metadata.

Direct native NBS refresh is optional and can be used as an additional cross-check when the host is reachable.

## Provenance policy

Every derived row should retain, directly or through its source index:

- canonical publisher/source dataset;
- upstream resource/file ID and URL;
- source worksheet where relevant;
- reference month;
- transport/mirror identifier where used;
- retrieval timestamp where available;
- processing code version;
- spatial aggregation level;
- transformations and unit normalization;
- missing-data/quality flags;
- required attribution.
