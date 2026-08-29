from __future__ import annotations

import pandas as pd


def temporal_split(df: pd.DataFrame, date_col: str, train_end: str, validation_end: str, test_start: str) -> pd.Series:
    dt = pd.to_datetime(df[date_col])
    train_end_ts = pd.Timestamp(train_end)
    validation_end_ts = pd.Timestamp(validation_end)
    test_start_ts = pd.Timestamp(test_start)
    if test_start_ts <= train_end_ts:
        raise ValueError("test_start must be after train_end")
    labels = pd.Series("gap", index=df.index, dtype="string")
    labels.loc[dt <= train_end_ts] = "train"
    labels.loc[(dt > train_end_ts) & (dt <= validation_end_ts)] = "validation"
    labels.loc[dt >= test_start_ts] = "test"
    return labels
