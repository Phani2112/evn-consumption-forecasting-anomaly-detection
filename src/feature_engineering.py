"""Lag and rolling feature engineering for the XGBoost stage."""

import pandas as pd

from src.config import LAG_PERIODS


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "consumption_d2",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Add lagged values and rolling means for *target_col*.

    Parameters
    ----------
    df:
        DataFrame sorted by timestamp (ascending).
    target_col:
        Column to lag.
    lags:
        Lag periods (in 15-min intervals).  Defaults to ``config.LAG_PERIODS``.

    Returns
    -------
    DataFrame with new lag and rolling-mean columns appended in-place copy.
    """
    if lags is None:
        lags = LAG_PERIODS

    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)

    df[f"{target_col}_roll4"] = df[target_col].rolling(4, min_periods=1).mean()
    df[f"{target_col}_roll24"] = df[target_col].rolling(24, min_periods=1).mean()

    return df
