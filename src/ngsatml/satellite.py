from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
from pyproj import Transformer

from .features import nbr, ndvi, ndwi
from .stac import CatalogQuery, search_items


def point_bbox(lon: float, lat: float, radius_m: float = 3000.0) -> tuple[float, float, float, float]:
    lat_delta = radius_m / 111_320.0
    lon_scale = max(0.1, math.cos(math.radians(lat)))
    lon_delta = radius_m / (111_320.0 * lon_scale)
    return lon - lon_delta, lat - lat_delta, lon + lon_delta, lat + lat_delta


def month_bounds(month: Any) -> tuple[str, str]:
    import pandas as pd

    start = pd.Timestamp(month).to_period("M").to_timestamp()
    end = start + pd.DateOffset(months=1)
    return start.date().isoformat(), end.date().isoformat()


def _valid_scene_item(item: dict) -> bool:
    assets = item.get("assets", {})
    return all(k in assets and isinstance(assets[k], dict) and assets[k].get("href") for k in ["red", "green", "nir", "swir16", "scl"])


def find_s2_scenes(
    lon: float,
    lat: float,
    month: Any,
    *,
    collection: str = "sentinel-2-l2a",
    cloud_cover_max: float = 70.0,
    query_radius_m: float = 3000.0,
    limit: int = 40,
) -> list[dict]:
    start, end = month_bounds(month)
    q = CatalogQuery(
        collection=collection,
        bbox=point_bbox(lon, lat, query_radius_m),
        start=start,
        end=end,
        cloud_cover_max=cloud_cover_max,
        limit=limit,
    )
    items = [i for i in search_items(q) if _valid_scene_item(i)]

    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    filtered = []
    for item in items:
        raw = item.get("properties", {}).get("datetime")
        try:
            observed = datetime.fromisoformat(str(raw).replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            observed = start_dt
        if not (start_dt <= observed < end_dt):
            continue
        cloud = item.get("properties", {}).get("eo:cloud_cover")
        cloud_value = float(cloud) if cloud is not None else 999.0
        filtered.append((cloud_value, abs((observed - (start_dt + (end_dt - start_dt) / 2)).total_seconds()), item))
    filtered.sort(key=lambda x: (x[0], x[1], str(x[2].get("id"))))
    return [x[2] for x in filtered]


def _read_patch(
    href: str,
    lon: float,
    lat: float,
    radius_m: float,
    out_size: int,
    *,
    categorical: bool = False,
) -> np.ma.MaskedArray:
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import from_bounds
    except ImportError as exc:
        raise RuntimeError("rasterio is required for satellite feature extraction") from exc

    env = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
        "GDAL_HTTP_MULTIRANGE": "YES",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    }
    with rasterio.Env(**env):
        with rasterio.open(href) as src:
            if src.crs is None:
                raise ValueError(f"Raster has no CRS: {href}")
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
            window = from_bounds(x - radius_m, y - radius_m, x + radius_m, y + radius_m, src.transform)
            return src.read(
                1,
                window=window,
                out_shape=(out_size, out_size),
                boundless=True,
                masked=True,
                fill_value=src.nodata if src.nodata is not None else 0,
                resampling=Resampling.nearest if categorical else Resampling.bilinear,
            ).astype("float32")


def _finite_values(array: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype="float32")[valid]
    return values[np.isfinite(values)]


def _stats(prefix: str, array: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    values = _finite_values(array, valid)
    if values.size == 0:
        return {f"{prefix}_{name}": float("nan") for name in ["mean", "median", "std", "p10", "p90"]}
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
    }


def _scale_reflectance(array: np.ndarray) -> np.ndarray:
    finite = np.asarray(array, dtype="float32")
    probe = finite[np.isfinite(finite)]
    if probe.size and float(np.nanmedian(np.abs(probe))) > 2.0:
        return finite / 10_000.0
    return finite


def extract_s2_scene_features(
    item: dict,
    lon: float,
    lat: float,
    *,
    patch_radius_m: float = 1280.0,
    out_size: int = 64,
) -> dict[str, Any]:
    assets = item["assets"]
    red = _read_patch(assets["red"]["href"], lon, lat, patch_radius_m, out_size)
    green = _read_patch(assets["green"]["href"], lon, lat, patch_radius_m, out_size)
    nir_band = _read_patch(assets["nir"]["href"], lon, lat, patch_radius_m, out_size)
    swir16 = _read_patch(assets["swir16"]["href"], lon, lat, patch_radius_m, out_size)
    scl = _read_patch(assets["scl"]["href"], lon, lat, patch_radius_m, out_size, categorical=True)

    masks = [np.ma.getmaskarray(x) for x in [red, green, nir_band, swir16, scl]]
    common = ~np.logical_or.reduce(masks)
    scl_values = np.asarray(scl.filled(0), dtype="int16")
    # SCL: retain dark pixels, vegetation, bare soil, water and unclassified;
    # reject no-data, saturated, cloud shadow, cloud/cirrus and snow/ice.
    clear = np.isin(scl_values, [2, 4, 5, 6, 7])
    valid = common & clear
    valid_count = int(valid.sum())
    if valid_count < max(16, int(out_size * out_size * 0.01)):
        raise ValueError(f"Insufficient clear pixels: {valid_count}/{out_size * out_size}")

    red_np = np.asarray(red.filled(np.nan), dtype="float32")
    green_np = np.asarray(green.filled(np.nan), dtype="float32")
    nir_np = np.asarray(nir_band.filled(np.nan), dtype="float32")
    swir_np = np.asarray(swir16.filled(np.nan), dtype="float32")

    ndvi_arr = ndvi(nir_np, red_np)
    ndwi_arr = ndwi(green_np, nir_np)
    nbr_arr = nbr(nir_np, swir_np)

    result: dict[str, Any] = {
        "s2_scene_id": item.get("id"),
        "s2_scene_datetime": item.get("properties", {}).get("datetime"),
        "s2_scene_cloud_cover": item.get("properties", {}).get("eo:cloud_cover"),
        "s2_clear_pixel_fraction": float(valid_count / (out_size * out_size)),
        "s2_patch_radius_m": float(patch_radius_m),
        "s2_patch_pixels": int(out_size),
    }
    result.update(_stats("s2_ndvi", ndvi_arr, valid))
    result.update(_stats("s2_ndwi", ndwi_arr, valid))
    result.update(_stats("s2_nbr", nbr_arr, valid))
    result.update(_stats("s2_red", _scale_reflectance(red_np), valid))
    result.update(_stats("s2_green", _scale_reflectance(green_np), valid))
    result.update(_stats("s2_nir", _scale_reflectance(nir_np), valid))
    result.update(_stats("s2_swir16", _scale_reflectance(swir_np), valid))
    return result


def extract_market_month_s2(
    lon: float,
    lat: float,
    month: Any,
    *,
    collection: str = "sentinel-2-l2a",
    cloud_cover_max: float = 70.0,
    patch_radius_m: float = 1280.0,
    out_size: int = 64,
    max_scenes: int = 3,
) -> dict[str, Any]:
    scenes = find_s2_scenes(
        lon,
        lat,
        month,
        collection=collection,
        cloud_cover_max=cloud_cover_max,
        query_radius_m=max(3000.0, patch_radius_m * 1.5),
    )
    errors: list[str] = []
    for item in scenes[:max_scenes]:
        try:
            features = extract_s2_scene_features(
                item,
                lon,
                lat,
                patch_radius_m=patch_radius_m,
                out_size=out_size,
            )
            features["s2_status"] = "ok"
            features["s2_candidate_scenes"] = len(scenes)
            return features
        except Exception as exc:
            errors.append(f"{item.get('id')}: {type(exc).__name__}: {exc}")
    return {
        "s2_status": "missing" if not scenes else "failed",
        "s2_candidate_scenes": len(scenes),
        "s2_error": " | ".join(errors)[:1500],
    }
