from __future__ import annotations

SOURCES = {
    "earth_search": {
        "url": "https://earth-search.aws.element84.com/v1",
        "datasets": ["sentinel-2-l2a", "sentinel-1-grd", "landsat-c2-l2", "cop-dem-glo-30"],
        "auth": "none for catalog",
    },
    "worldcover": {
        "url": "https://planetarycomputer.microsoft.com/api/stac/v1/",
        "dataset": "esa-worldcover",
        "license": "CC-BY-4.0",
    },
    "geoboundaries": {
        "url": "https://www.geoboundaries.org/api/current/gbOpen/NGA/ADM1/",
        "license": "CC-BY-4.0",
    },
    "wfp_food_prices": {
        "url": "https://data.humdata.org/dataset/wfp-food-prices-for-nigeria",
        "role": "market x month x commodity price supervision",
    },
    "nbs_food_cpi": {
        "url": "https://microdata.nigerianstat.gov.ng/index.php/catalog/154/related-materials",
        "dataset": "NGA-NBS-CPI",
        "role": "36 states + FCT monthly Food CPI target/context layer",
    },
    "nbs_cohd": {
        "url": "https://microdata.nigerianstat.gov.ng/index.php/catalog/146/related-materials",
        "dataset": "NGA-NBS-COHD",
        "role": "36 states + FCT monthly Cost of a Healthy Diet target/context layer",
    },
    "nbs_selected_food_price_watch": {
        "url": "https://microdata.nigerianstat.gov.ng/index.php/catalog/162/related-materials",
        "dataset": "NGA-NBS-FOODPW",
        "role": "national/zonal commodity validation; not assumed to be a complete 37-state commodity label matrix",
    },
    "nbs_hf_transport": {
        "url": "https://huggingface.co/electricsheepafrica",
        "role": "provenance-preserving Parquet transport mirror for NBS CPI/CoHD when the NADA host is unavailable",
        "canonical_publisher": "National Bureau of Statistics, Nigeria",
    },
    "imerg": {
        "url": "https://gpm.nasa.gov/data/directory",
        "version": "V07",
        "auth": "free NASA Earthdata/PPS account may be required",
    },
}
