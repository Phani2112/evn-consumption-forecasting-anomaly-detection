"""Data loading, cleaning, and feature engineering for EVN smart-meter data."""

import pandas as pd

from src.config import FILE_PATH, FILTER_YEAR, SHEET_NAMES


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw_sheets(file_path: str = FILE_PATH) -> dict[str, pd.DataFrame]:
    """Load all five Excel sheets and return them in a dict keyed by short name."""
    xls = pd.ExcelFile(file_path)
    return {
        key: pd.read_excel(xls, sheet_name=name)
        for key, name in SHEET_NAMES.items()
    }


# ---------------------------------------------------------------------------
# Column renaming
# ---------------------------------------------------------------------------

def rename_columns(
    df_target: pd.DataFrame,
    df_plaus: pd.DataFrame,
    df_smart: pd.DataFrame,
    df_pv: pd.DataFrame,
    df_weather: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rename verbose column headers to short identifiers."""
    df_target = df_target.rename(columns={
        "Timestamp": "timestamp",
        "Target value (EVN KG Smart Meter IME Consumption ek1 Consumption per ZP "
        "(ground truth)) kWh": "consumption_gt",
        "Target value (Parameter1_EVN KG Smart Meter IME Consumption ek1 Consumption "
        "per ZP (d+2 at 3 PM)) kWh": "consumption_d2",
        "Target value (Parameter2_EVN KG Smart Meter IME Consumption ek1 Consumption "
        "per ZP (d+3 at 3 PM)) kWh": "consumption_d3",
    })

    df_plaus = df_plaus.rename(columns={
        "Timestamp": "timestamp",
        "Plausibility check (Plausibility check of target variable (d+2) in "
        "15-minute intervals (1 plausible, -1 implausible)) without": "plaus_d2_15m",
        "Plausibility check (plausibility check of target variable (d+3) in a "
        "15-minute grid (1 plausible, -1 implausible)) without": "plaus_d3_15m",
    })

    df_smart = df_smart.rename(columns={
        "Timestamp": "timestamp",
        "Inputs (EVN KG Smart Meter IME Feed-in ek2 Feed-in per ZP "
        "(d+2 at 3 PM)) kWh": "feedin_d2",
        "Inputs (EVN KG Smart Meter IME Feed-in ek2 Feed-in per ZP "
        "(d+3 at 3 PM)) kWh": "feedin_d3",
        "Inputs (EVN KG Smart Meter IME Consumption ek1 Consumption measured "
        "(d+2 at 3 PM)) kWh": "measured_d2",
        "Inputs (EVN KG Smart Meter IME Consumption ek1 Consumption measured "
        "(d+3 at 3 PM)) kWh": "measured_d3",
    })

    df_pv = df_pv.rename(columns={
        "Timestamp": "timestamp",
        "Inputs (total PV generation in Lower Austria combined (d+2 at 3 p.m.))kWh": "pv_d2",
        "Inputs (total PV generation in Lower Austria combined (d+3 at 3 p.m.))kWh": "pv_d3",
    })

    df_weather.columns = df_weather.columns.str.strip()
    df_weather = df_weather.rename(columns={
        "Timestamp": "timestamp",
        "Inputs (Temperature combination (d+2 at 3 p.m.))°C": "temp_d2",
        "Inputs (Temperature combination (d+3 at 3 p.m.))°C": "temp_d3",
        "Inputs (Global radiation combination (d+2 at 3 p.m.))W/m²": "rad_d2",
        "Inputs (global radiation combination (d+3 at 3 p.m.))W/m²": "rad_d3",
    })

    return df_target, df_plaus, df_smart, df_pv, df_weather


# ---------------------------------------------------------------------------
# Timestamp conversion
# ---------------------------------------------------------------------------

def convert_timestamps(*dfs: pd.DataFrame) -> list[pd.DataFrame]:
    """Convert the 'timestamp' column to datetime in each dataframe."""
    result = []
    for df in dfs:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        result.append(df)
    return result


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_to_15min(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Deduplicate by timestamp, resample to 15-min grid with forward-fill."""
    df = df.groupby(timestamp_col).mean(numeric_only=True)
    df = df.resample("15min").ffill().reset_index()
    return df


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def merge_datasets(
    df_target: pd.DataFrame,
    df_plaus: pd.DataFrame,
    df_smart: pd.DataFrame,
    df_pv_15m: pd.DataFrame,
    df_weather_15m: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join all five datasets on 'timestamp'."""
    df = (
        df_target
        .merge(df_plaus, on="timestamp", how="left")
        .merge(df_smart, on="timestamp", how="left")
        .merge(df_pv_15m, on="timestamp", how="left")
        .merge(df_weather_15m, on="timestamp", how="left")
    )
    return df


# ---------------------------------------------------------------------------
# Filtering & sorting
# ---------------------------------------------------------------------------

def filter_year(df: pd.DataFrame, year: int = FILTER_YEAR) -> pd.DataFrame:
    """Keep only rows from the specified calendar year."""
    return df[df["timestamp"].dt.year == year].copy()


def sort_by_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by timestamp and reset the index."""
    return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Time feature engineering
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features and the D3-D2 delta column."""
    df = df.copy()
    df["delta_d3_d2"] = df["consumption_d3"] - df["consumption_d2"]
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_dataset(file_path: str = FILE_PATH) -> pd.DataFrame:
    """End-to-end: load → rename → convert → resample → merge → filter → engineer.

    Returns a clean, sorted DataFrame ready for modelling.
    """
    sheets = load_raw_sheets(file_path)
    df_target, df_plaus, df_smart, df_pv, df_weather = rename_columns(
        sheets["target"],
        sheets["plaus"],
        sheets["smart"],
        sheets["pv"],
        sheets["weather"],
    )

    df_target, df_plaus, df_smart, df_pv, df_weather = convert_timestamps(
        df_target, df_plaus, df_smart, df_pv, df_weather
    )

    df_pv_15m = resample_to_15min(df_pv)
    df_weather_15m = resample_to_15min(df_weather)

    df = merge_datasets(df_target, df_plaus, df_smart, df_pv_15m, df_weather_15m)
    df = filter_year(df)
    df = sort_by_timestamp(df)
    df = add_time_features(df)
    return df
