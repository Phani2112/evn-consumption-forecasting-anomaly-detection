"""Anomaly detection methods for EVN consumption residuals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import IQR_MULT_RF, IQR_MULT_XGB, MIN_CONSECUTIVE, WINDOW_SIZE, Z_THRESH


# ---------------------------------------------------------------------------
# Residual helpers
# ---------------------------------------------------------------------------

def compute_residuals(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> pd.Series:
    """Return signed residuals (y_true − y_pred) as a Series."""
    return pd.Series(np.asarray(y_true) - np.asarray(y_pred))


# ---------------------------------------------------------------------------
# Z-score anomaly detector
# ---------------------------------------------------------------------------

def detect_anomaly_zscore(
    residuals: pd.Series,
    window: int = WINDOW_SIZE,
    threshold: float = Z_THRESH,
) -> pd.Series:
    """Flag anomalies using a causal rolling Z-score.

    Parameters
    ----------
    residuals:
        Signed residual series (chronologically ordered).
    window:
        Rolling window size (number of 15-min intervals).
    threshold:
        Absolute Z-score threshold above which a point is flagged.

    Returns
    -------
    Boolean Series (``True`` = anomaly, ``pd.NA`` where Z-score is unavailable).
    """
    res_mean = residuals.rolling(window, min_periods=window).mean()
    res_std = residuals.rolling(window, min_periods=window).std()
    res_std_safe = res_std.replace(0, np.nan)

    z = (residuals - res_mean) / res_std_safe
    anomaly = (z.abs() > threshold).astype("boolean")
    anomaly[z.isna()] = pd.NA
    return anomaly


# ---------------------------------------------------------------------------
# IQR anomaly detector
# ---------------------------------------------------------------------------

def iqr_threshold(
    train_abs_residuals: np.ndarray | pd.Series,
    multiplier: float = IQR_MULT_RF,
) -> float:
    """Compute Q3 + multiplier × IQR from training absolute residuals."""
    arr = np.asarray(train_abs_residuals)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    return float(q3 + multiplier * (q3 - q1))


def detect_anomaly_iqr(
    abs_residuals: pd.Series,
    threshold: float,
) -> pd.Series:
    """Flag anomalies where *abs_residuals* exceed *threshold*.

    Parameters
    ----------
    abs_residuals:
        Absolute residual series on the test set.
    threshold:
        Pre-computed IQR threshold (from training data only).

    Returns
    -------
    Boolean Series (``True`` = anomaly, ``pd.NA`` where residual is NaN).
    """
    anomaly = (abs_residuals > threshold).astype("boolean")
    anomaly[abs_residuals.isna()] = pd.NA
    return anomaly


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def remove_isolated(
    series: pd.Series,
    min_consecutive: int = MIN_CONSECUTIVE,
) -> pd.Series:
    """Remove isolated anomaly flags; keep only runs of length ≥ *min_consecutive*.

    Parameters
    ----------
    series:
        Boolean anomaly flag series (NaN treated as False via caller's ``fillna``).
    min_consecutive:
        Minimum number of consecutive True values required to keep a flag.

    Returns
    -------
    Cleaned boolean Series.
    """
    s = series.astype(bool)
    groups = (s != s.shift()).cumsum()
    run_lengths = s.groupby(groups).transform("sum")
    return s & (run_lengths >= min_consecutive)


# ---------------------------------------------------------------------------
# XGBoost-specific anomaly detection
# ---------------------------------------------------------------------------

def detect_anomaly_xgb(
    train_abs_residuals: np.ndarray | pd.Series,
    test_abs_residuals: pd.Series,
    multiplier: float = IQR_MULT_XGB,
) -> tuple[pd.Series, float]:
    """IQR-based anomaly detection tuned for XGBoost residuals.

    Computes the threshold from *train_abs_residuals* and applies it to
    *test_abs_residuals*.

    Returns
    -------
    anomaly_flags : integer Series (1 = anomaly, 0 = normal)
    threshold : float
    """
    thresh = iqr_threshold(train_abs_residuals, multiplier=multiplier)
    flags = (test_abs_residuals > thresh).astype(int)
    return flags, thresh
