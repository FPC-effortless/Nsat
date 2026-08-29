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
    "nbs_food_prices": {
        "url": "https://microdata.nigerianstat.gov.ng/index.php/catalog/162/related-materials",
        "dataset": "NGA-NBS-FOODPW",
    },
    "imerg": {
        "url": "https://gpm.nasa.gov/data/directory",
        "version": "V07",
        "auth": "free NASA Earthdata/PPS account may be required",
    },
}
