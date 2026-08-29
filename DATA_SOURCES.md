# Data source registry

This file is operational guidance, not legal advice. Re-check source terms before commercial release.

| Source | Role | Access | License / use note | Auth |
|---|---|---|---|---|
| Copernicus Sentinel-2 L2A | Optical imagery | Earth Search / Copernicus Data Space STAC | Sentinel data are free, full and open | Earth Search: none for catalog |
| Copernicus Sentinel-1 GRD | Radar imagery | Earth Search / AWS Open Data | Sentinel data are free, full and open | none for Earth Search catalog |
| Copernicus DEM GLO-30 | Elevation | Earth Search | Verify Copernicus DEM notice/attribution | none for Earth Search catalog |
| ESA WorldCover 2021 | Cropland/land-cover mask | AWS Open Data or Planetary Computer STAC | CC BY 4.0 | none for AWS; Planetary Computer public STAC |
| geoBoundaries gbOpen NGA ADM1 | Nigeria state boundaries | geoBoundaries API | CC BY 4.0 | none |
| NASA GPM IMERG V07 | Rainfall | NASA GES DISC/PPS | NASA-led mission data are generally open; verify product-specific terms and cite source | free Earthdata/PPS account may be required |
| NBS Selected Food Price Watch | Supervised targets | NBS Microdata Catalog / eLibrary | Re-check NBS publication terms before redistribution; preserve provenance | usually public download |

## Canonical endpoints

- Earth Search STAC: `https://earth-search.aws.element84.com/v1`
- Copernicus Data Space STAC: `https://stac.dataspace.copernicus.eu/v1/`
- Planetary Computer STAC: `https://planetarycomputer.microsoft.com/api/stac/v1/`
- geoBoundaries NGA ADM1 API: `https://www.geoboundaries.org/api/current/gbOpen/NGA/ADM1/`
- NBS Selected Food Price Watch catalog: `https://microdata.nigerianstat.gov.ng/index.php/catalog/162/related-materials`
- NASA IMERG directory: `https://gpm.nasa.gov/data/directory`

## Provenance policy

Every derived row should retain source dataset/version, source item/file identifier, reference date, processing code version, spatial aggregation level, transformations, missing-data flags, and required attribution.
