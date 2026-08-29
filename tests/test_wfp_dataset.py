import pandas as pd

from ngsatml.satellite import month_bounds, point_bbox
from ngsatml.wfp import add_calendar_targets, normalize_prices, select_market_months


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


def test_calendar_targets_do_not_bridge_missing_months():
    labels = normalize_prices(_raw_prices())
    out = add_calendar_targets(labels, (1, 3))
    jan = out[out["month"].eq(pd.Timestamp("2025-01-01"))].iloc[0]
    april = out[out["month"].eq(pd.Timestamp("2025-04-01"))].iloc[0]
    assert jan["target_price_ngn_1m"] == 120.0
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


def test_satellite_query_helpers_are_month_and_location_bounded():
    assert month_bounds("2025-01-15") == ("2025-01-01", "2025-02-01")
    west, south, east, north = point_bbox(7.24, 13.08, 1000)
    assert west < 7.24 < east
    assert south < 13.08 < north
