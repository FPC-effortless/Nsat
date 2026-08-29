import pandas as pd

from ngsatml.quality import add_quality_columns, _state_weights


def _frame():
    return pd.DataFrame(
        {
            "month": ["2020-11-01", "2021-06-01", "2022-06-01", "2023-06-01"],
            "target_month": ["2020-12-01", "2021-07-01", "2022-07-01", "2023-07-01"],
            "price_ngn": [100.0, 100.0, 100.0, 100.0],
            "target_price_ngn_1m": [110.0, 700.0, 90.0, 105.0],
            "price_lag_1m": [95.0, 100.0, 10.0, 100.0],
            "lag_feature_count": [3, 3, 3, 1],
            "admin1": ["A", "A", "B", "B"],
        }
    )


def test_quality_columns_create_national_and_ood_regimes():
    out = add_quality_columns(_frame())
    assert out["split_national"].tolist() == ["train", "validation", "test", "recent_ood"]
    assert out["national_target_within_split"].all()
    assert bool(out.loc[1, "quality_extreme_target_change"])
    assert bool(out.loc[2, "quality_extreme_input_jump_1m"])
    assert bool(out.loc[3, "quality_sparse_history"])


def test_cross_boundary_target_is_not_within_national_split():
    frame = _frame().head(1).copy()
    frame.loc[0, "month"] = "2020-12-01"
    frame.loc[0, "target_month"] = "2021-01-01"
    out = add_quality_columns(frame)
    assert out.loc[0, "split_national"] == "train"
    assert out.loc[0, "target_split_national"] == "validation"
    assert not bool(out.loc[0, "national_target_within_split"])


def test_state_weights_are_bounded():
    train = pd.DataFrame({"admin1": ["A"] * 100 + ["B"] * 10 + ["C"]})
    weights = _state_weights(train)
    assert all(0.5 <= value <= 5.0 for value in weights.values())
    assert weights["A"] < weights["B"] < weights["C"]
