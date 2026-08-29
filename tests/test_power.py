import pandas as pd

from ngsatml.power import _parse_power_payload


def test_parse_power_payload_creates_monthly_features_and_precip_total():
    payload = {
        "properties": {
            "parameter": {
                "PRECTOTCORR": {"202401": 2.0, "202402": 3.0, "202413": 2.5},
                "T2M": {"202401": 25.0, "202402": 26.0, "202413": 25.5},
                "RH2M": {"202401": 40.0, "202402": -999.0, "202413": 45.0},
            }
        }
    }
    frame = _parse_power_payload(payload, market_id=123)
    assert frame["month"].tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")]
    jan = frame.iloc[0]
    feb = frame.iloc[1]
    assert jan["market_id"] == 123
    assert jan["power_t2m"] == 25.0
    assert jan["power_precip_est_mm_month"] == 62.0
    assert pd.isna(feb["power_rh2m"])
    assert feb["power_precip_est_mm_month"] == 87.0
