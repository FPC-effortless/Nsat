import pandas as pd

from ngsatml.satellite import month_bounds, point_bbox
from ngsatml.wfp import add_calendar_targets, normalize_prices, normalize_unit, select_market_months


def _raw_prices():
    return pd.DataFrame(
        [
            ["2025-01-15", "Katsina", "Jibia", "Jibia", 1038, 13.08, 7.24, "Maize", "KG", "Wholesale", "NGN", 100.0, 0.1],
            ["2025-02-15", "Katsina", "Jibia", "Jibia", 1038, 13.08, 7.24, "Maize", "KG", "Wholesale", "NGN", 120.0, 0.12],
            ["2025-04-15", "Katsina", "Jibia", "Jibia", 1038, 13.08, 7.24, "Maize", "KG", "Wholesale", "NGN", 180.0, 0.18],
        ],
        columns=[
            "date", "admin1", "admin2", "market", "market_id", "latitude", "longitude",
            "commodity", "unit", "pricetype", "currency", "price", "usdprice",
        ],
    )


def test_unit_normalization_converts_packages_to_base_units():
    assert normalize_unit("KG") == ("kg", 1.0, "mass")
    assert normalize_unit("500 G") == ("kg", 0.5, "mass")
    assert normalize_unit("2.5 KG") == ("kg", 2.5, "mass")
    assert normalize_unit("30 pcs") == ("item", 30.0, "count")
    assert normalize_unit("100 Tubers") == ("item", 100.0, "count")

    raw = pd.DataFrame(
        [
            ["2025-01-15", "Katsina", "Jibia", "Jibia", 1038, 13.08, 7.24, "Maize", "500 G", "Retail", "NGN", 500.0, 0.3],
            ["2025-01-16", "Katsina", "Jibia", "Jibia", 1038, 13.08, 7.24, "Maize", "2 KG", "Retail", "NGN", 2000.0, 1.2],
        ],
        columns=[
            "date", "admin1", "admin2", "market", "market_id", "latitude", "longitude",
            "commodity", "unit", "pricetype", "currency", "price", "usdprice",
        ],
    )
    labels = normalize_prices(raw, units=None)
    assert len(labels) == 1
    assert labels.iloc[0]["unit"] == "kg"
    assert labels.iloc[0]["price_ngn"] == 1000.0
    assert labels.iloc[0]["source_unit_count"] == 2


def test_calendar_targets_do_not_bridge_missing_months():
    labels = normalize_prices(_raw_prices())
    out = add_calendar_targets(labels, (1, 3))
    jan = out[out["month"].eq(pd.Timestamp("2025-01-01"))].iloc[0]
    april = out[out["month"].eq(pd.Timestamp("2025-04-01"))].iloc[0]
    assert jan["target_price_ngn_1m"] == 120.0
    assert jan["target_month"] == pd.Timestamp("2025-02-01")
    assert jan["price_lag_1m"] != jan["price_lag_1m"]
    assert april["price_lag_3m"] == 100.0
    assert april["price_lag_1m"] != april["price_lag_1m"]


def test_select_market_month_requires_real_next_month_target():
    labels = add_calendar_targets(normalize_prices(_raw_prices()), (1,))
    selected = select_market_months(
        labels,
        start_date="2025-01-01",
        end_date="2025-05-01",
        require_next_target=True,
    )
    assert selected["month"].tolist() == [pd.Timestamp("2025-01-01")]


def test_temporal_spread_sampling_uses_distinct_months():
    rows = []
    for month in range(1, 7):
        rows.append(
            [
                f"2024-{month:02d}-15", "Katsina", "Jibia", "Jibia", 1038, 13.08, 7.24,
                "Maize", "KG", "Wholesale", "NGN", 100.0 + month, 0.1,
            ]
        )
    raw = pd.DataFrame(
        rows,
        columns=[
            "date", "admin1", "admin2", "market", "market_id", "latitude", "longitude",
            "commodity", "unit", "pricetype", "currency", "price", "usdprice",
        ],
    )
    labels = add_calendar_targets(normalize_prices(raw), (1,))
    selected = select_market_months(
        labels,
        start_date="2024-01-01",
        end_date="2024-06-01",
        limit=3,
        require_next_target=True,
        spread_across_months=True,
    )
    assert len(selected) == 3
    assert selected["month"].nunique() == 3
    assert selected["month"].min() == pd.Timestamp("2024-01-01")
    assert selected["month"].max() == pd.Timestamp("2024-05-01")


def test_spatial_temporal_sampler_uses_multiple_states():
    rows = []
    states = [("Katsina", 1001, 13.0, 7.0), ("Borno", 1002, 12.0, 13.0), ("Abia", 1003, 5.1, 7.3)]
    for month in range(1, 5):
        for state, market_id, lat, lon in states:
            for offset in (0, 1):
                rows.append(
                    [
                        f"2024-{month:02d}-15", state, "LGA", f"{state} market", market_id,
                        lat, lon, "Maize", "KG", "Retail", "NGN", 100 + month + offset, 0.1,
                    ]
                )
    raw = pd.DataFrame(
        rows,
        columns=[
            "date", "admin1", "admin2", "market", "market_id", "latitude", "longitude",
            "commodity", "unit", "pricetype", "currency", "price", "usdprice",
        ],
    )
    labels = add_calendar_targets(normalize_prices(raw), (1,))
    selected = select_market_months(
        labels,
        start_date="2024-01-01",
        end_date="2024-04-01",
        limit=6,
        require_next_target=True,
        spread_across_states=True,
    )
    assert len(selected) == 6
    assert selected["admin1"].nunique() == 3
    assert selected["month"].nunique() == 3


def test_satellite_query_helpers_are_month_and_location_bounded():
    assert month_bounds("2025-01-15") == ("2025-01-01", "2025-02-01")
    west, south, east, north = point_bbox(7.24, 13.08, 1000)
    assert west < 7.24 < east
    assert south < 13.08 < north
