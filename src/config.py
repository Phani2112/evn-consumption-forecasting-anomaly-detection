"""Project-wide constants and hyper-parameters."""

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
FILE_PATH = "Data_Doctor_Daten_Translated.xlsx"

SHEET_NAMES = {
    "target": "Target size_15min",
    "plaus": "Plausibility check_15 min",
    "smart": "Inputs_IME_Smart_Meter",
    "pv": "Inputs_PV",
    "weather": "Inputs_Weather",
}

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
TARGET = "consumption_gt"

FEATURES = [
    "consumption_d2",
    "consumption_d3",
    "delta_d3_d2",
    "feedin_d2",
    "feedin_d3",
    "measured_d2",
    "measured_d3",
    "pv_d2",
    "pv_d3",
    "temp_d2",
    "temp_d3",
    "rad_d2",
    "rad_d3",
    "hour",
    "dayofweek",
    "month",
]

# XGBoost feature set (adds lag/rolling features, drops consumption_d3)
FEATURES_XGB = [
    "consumption_d2",
    "delta_d3_d2",
    "measured_d2",
    "temp_d2",
    "temp_d3",
    "rad_d2",
    "rad_d3",
    "pv_d2",
    "pv_d3",
    "feedin_d2",
    "feedin_d3",
    "hour",
    "dayofweek",
    "month",
    "consumption_d2_lag1",
    "consumption_d2_lag2",
    "consumption_d2_lag4",
    "consumption_d2_lag24",
    "consumption_d2_roll4",
    "consumption_d2_roll24",
]

TRAIN_TEST_SPLIT = 0.8
FILTER_YEAR = 2025

# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------
RF_PARAMS = {
    "n_estimators": 200,
    "random_state": 42,
    "n_jobs": -1,
}

# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 5,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "min_child_weight": 2,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.5,
    "reg_lambda": 1.0,
    "reg_alpha": 0.5,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

LAG_PERIODS = [1, 2, 4, 24]

# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------
WINDOW_SIZE = 96        # 24 h × 4 (15-min intervals)
Z_THRESH = 3.0
IQR_MULT_RF = 1.5       # multiplier for RF-based anomaly detection
IQR_MULT_XGB = 2.5      # multiplier for XGBoost-based anomaly detection
MIN_CONSECUTIVE = 2     # minimum consecutive anomaly flags to keep
