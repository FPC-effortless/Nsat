import numpy as np
import pandas as pd

from ngsatml.features import aggregate_state_month, ndvi


def test_ndvi_known_values():
    nir = np.array([0.8, 0.2])
    red = np.array([0.2, 0.2])
    result = ndvi(nir, red)
    assert np.isclose(result[0], 0.6, atol=1e-5)
    assert np.isclose(result[1], 0.0, atol=1e-5)


def test_aggregate_is_state_month_not_patch_label():
    df = pd.DataFrame({
        "state": ["Kano", "Kano", "Benue"],
        "date": ["2025-01-02", "2025-01-20", "2025-01-03"],
        "ndvi": [0.2, 0.6, 0.7],
    })
    out = aggregate_state_month(df)
    assert len(out) == 2
    kano = out[out["state"] == "Kano"].iloc[0]
    assert np.isclose(kano["ndvi_median"], 0.4)
