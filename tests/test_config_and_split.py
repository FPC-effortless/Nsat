import pandas as pd

from ngsatml.splits import temporal_split


def test_temporal_split():
    df = pd.DataFrame({"date": ["2024-12-01", "2025-06-01", "2026-02-01"]})
    labels = temporal_split(df, "date", "2024-12-31", "2025-12-31", "2026-01-01")
    assert labels.tolist() == ["train", "validation", "test"]
