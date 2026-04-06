"""Evaluation utilities: metrics, confusion matrices, and plotting helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


# ---------------------------------------------------------------------------
# Anomaly vs plausibility evaluation
# ---------------------------------------------------------------------------

def eval_vs_plaus(
    df: pd.DataFrame,
    flag_col: str,
    plaus_col: str = "plaus_d2_15m",
    name: str = "",
    print_results: bool = True,
) -> dict:
    """Evaluate an anomaly detector against EVN plausibility labels.

    EVN coding: plausibility == -1  ⟹  implausible (positive class).

    Parameters
    ----------
    df:
        Test DataFrame containing *flag_col* and *plaus_col*.
    flag_col:
        Column with boolean or integer anomaly flags (True / 1 = anomaly).
    plaus_col:
        Plausibility column name.
    name:
        Display name shown in printed output.
    print_results:
        Whether to print confusion matrix and classification report.

    Returns
    -------
    dict with keys ``confusion_matrix``, ``classification_report``,
    ``precision``, ``recall``, ``f1``.
    """
    valid = df.dropna(subset=[flag_col, plaus_col]).copy()
    y_true = (valid[plaus_col] == -1).astype(int)
    y_pred = valid[flag_col].fillna(False).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=3, output_dict=True)
    report_str = classification_report(y_true, y_pred, digits=3)

    if print_results:
        header = f"=== {name} vs plausibility ===" if name else "=== Anomaly vs plausibility ==="
        print(f"\n{header}")
        print("Rows = plausibility (0 plausible, 1 implausible)")
        print("Cols = detector     (0 normal,   1 anomaly)")
        print(cm)
        print("\nReport:")
        print(report_str)

    precision = report.get("1", {}).get("precision", float("nan"))
    recall = report.get("1", {}).get("recall", float("nan"))
    f1 = report.get("1", {}).get("f1-score", float("nan"))

    return {
        "confusion_matrix": cm,
        "classification_report": report_str,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Plotting helpers (require matplotlib)
# ---------------------------------------------------------------------------

def plot_mae_comparison(
    labels: list[str],
    mae_values: list[float],
    title: str = "MAE Comparison",
) -> None:
    """Bar chart comparing MAE across methods."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, mae_values)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
        )
    plt.ylabel("MAE (kWh)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_error_over_time(
    timestamps: pd.Series,
    errors: pd.Series,
    title: str = "Absolute Prediction Error Over Time",
) -> None:
    """Line plot of absolute prediction error with a marked maximum."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 4))
    plt.plot(timestamps, errors, linewidth=0.7, label="|GT − Predicted GT|")

    max_idx = errors.idxmax()
    plt.scatter(
        timestamps.loc[max_idx],
        errors.loc[max_idx],
        color="red",
        label=f"Max = {errors.loc[max_idx]:.4f}",
    )
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Absolute Error (kWh)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_anomalies_on_residuals(
    df: pd.DataFrame,
    residual_col: str,
    anomaly_col: str,
    timestamp_col: str = "timestamp",
    title: str = "Anomalies on Residual Signal",
) -> None:
    """Overlay anomaly scatter-points on the residual time-series."""
    import matplotlib.pyplot as plt

    tmp = df.dropna(subset=[anomaly_col]).copy()

    plt.figure(figsize=(14, 4))
    plt.plot(tmp[timestamp_col], tmp[residual_col], linewidth=0.6, label=residual_col)
    flagged = tmp[tmp[anomaly_col].astype(bool)]
    plt.scatter(
        flagged[timestamp_col],
        flagged[residual_col],
        s=10,
        label="Anomaly",
    )
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Absolute Residual (kWh)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def feature_importance_report(
    model,
    feature_names: list[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """Return a sorted feature-importance DataFrame."""
    df = pd.DataFrame(
        {"Feature": feature_names, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)
    return df.head(top_n).reset_index(drop=True)
