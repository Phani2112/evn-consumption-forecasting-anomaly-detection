"""Training and evaluation helpers for Random Forest and XGBoost forecasters."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.config import FEATURES, FEATURES_XGB, RF_PARAMS, TARGET, TRAIN_TEST_SPLIT, XGB_PARAMS
from src.feature_engineering import add_lag_features


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def time_based_split(
    df: pd.DataFrame,
    features: list[str],
    target: str = TARGET,
    split_ratio: float = TRAIN_TEST_SPLIT,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, int]:
    """Chronological 80/20 split (no shuffle).

    Returns
    -------
    X_train, y_train, X_test, y_test, split_index
    """
    df_model = df.dropna(subset=[target]).copy()
    split_index = int(len(df_model) * split_ratio)

    X = df_model[features]
    y = df_model[target]

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, y_train, X_test, y_test, split_index


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict | None = None,
) -> RandomForestRegressor:
    """Fit a Random Forest regressor and return the trained model."""
    if params is None:
        params = RF_PARAMS
    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)
    return rf


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict | None = None,
) -> XGBRegressor:
    """Fit an XGBoost regressor and return the trained model."""
    if params is None:
        params = XGB_PARAMS
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_forecast(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Return MAE, RMSE, R², and safe MAPE."""
    y_true = np.asarray(y_true)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    mask = np.abs(y_true) > eps
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def compare_baselines(
    df_test: pd.DataFrame,
    y_pred: np.ndarray,
    target: str = TARGET,
) -> pd.DataFrame:
    """Compare model MAE against D+2 and D+3 raw baselines.

    Parameters
    ----------
    df_test:
        Test-set slice of df_model (must contain *target*, ``consumption_d2``,
        and ``consumption_d3`` columns).
    y_pred:
        Model predictions aligned with *df_test*.
    target:
        Ground-truth column name.

    Returns
    -------
    DataFrame with columns ``method`` and ``MAE``.
    """
    rows = [
        ("D+2 baseline", mean_absolute_error(df_test[target], df_test["consumption_d2"])),
        ("D+3 baseline", mean_absolute_error(df_test[target], df_test["consumption_d3"])),
        ("Model", mean_absolute_error(df_test[target], y_pred)),
    ]
    return pd.DataFrame(rows, columns=["method", "MAE"])


# ---------------------------------------------------------------------------
# Convenience: build the XGBoost dataset (adds lags, splits)
# ---------------------------------------------------------------------------

def build_xgb_dataset(
    df_model: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target: str = TARGET,
    split_ratio: float = TRAIN_TEST_SPLIT,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Add lag features and return X/y train/test for the XGBoost model.

    Returns
    -------
    X_train_xgb, y_train_xgb, X_test_xgb, y_test_xgb
    """
    if feature_cols is None:
        feature_cols = FEATURES_XGB

    df_xgb = df_model.copy()
    df_xgb["timestamp"] = pd.to_datetime(df_xgb["timestamp"])
    df_xgb = add_lag_features(df_xgb, "consumption_d2")
    df_xgb = add_lag_features(df_xgb, "consumption_d3")

    split_idx = int(len(df_xgb) * split_ratio)
    split_date = df_xgb.iloc[split_idx]["timestamp"]

    df_train = df_xgb[df_xgb["timestamp"] < split_date].copy()
    df_test = df_xgb[df_xgb["timestamp"] >= split_date].copy()

    X_train = df_train[feature_cols].dropna()
    y_train = df_train.loc[X_train.index, target]

    X_test = df_test[feature_cols].dropna()
    y_test = df_test.loc[X_test.index, target]

    return X_train, y_train, X_test, y_test
