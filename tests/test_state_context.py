from __future__ import annotations

import pandas as pd

from ngsatml.state_context import attach_nbs_state_context


def test_attach_nbs_state_context_keeps_only_predictor_columns():
    markets = pd.DataFrame([
        {"admin1": "Nassarawa", "month": "2026-03-01", "commodity": "Maize"},
        {"admin1": "Nasarawa", "month": "2026-03-01", "commodity": "Rice"},
    ])
    context = pd.DataFrame([
        {
            "state": "Nasarawa",
            "month": "2026-03-01",
            "food_cpi": 140.0,
            "index_regime": "2024-base",
            "food_cpi_lag_1m": 137.0,
            "target_food_cpi_1m": 143.0,
            "cohd_ngn_person_day": 1900.0,
            "cohd_lag_1m": 1850.0,
            "target_cohd_1m": 1950.0,
        }
    ])
    result = attach_nbs_state_context(markets, context)
    assert len(result) == 2
    assert result["food_cpi"].eq(140.0).all()
    assert result["cohd_ngn_person_day"].eq(1900.0).all()
    assert result["nbs_state_context_available"].all()
    assert "target_food_cpi_1m" not in result.columns
    assert "target_cohd_1m" not in result.columns


def test_attach_nbs_state_context_requires_unique_state_month():
    markets = pd.DataFrame([{"admin1": "Kano", "month": "2026-03-01"}])
    context = pd.DataFrame([
        {"state": "Kano", "month": "2026-03-01", "food_cpi": 1.0},
        {"state": "Kano", "month": "2026-03-01", "food_cpi": 2.0},
    ])
    try:
        attach_nbs_state_context(markets, context)
    except ValueError as exc:
        assert "not unique" in str(exc)
    else:
        raise AssertionError("duplicate state-month context must be rejected")
