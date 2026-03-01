# Energy Consumption Forecasting & Anomaly Detection
### EVN Smart Meter Data | CRISP-DM Methodology

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green)
![Methodology](https://img.shields.io/badge/Methodology-CRISP--DM-purple)

---

## Business Problem

EVN, a major Austrian energy utility, must plan power plant operations 
and energy purchases based on customer consumption data. However, the 
earliest available meter readings (D+2 snapshots) are noisy and 
unreliable — the final ground truth only arrives days later. 

This project addresses two core challenges:
1. **Forecasting:** Predict final consumption as early as possible
2. **Anomaly Detection:** Automatically flag suspicious readings 
   before they impact operational decisions

---

## Solution: Two-Stage Hybrid ML Pipeline


### Stage 1 — Random Forest Forecaster
- Predicts final ground truth consumption from early D+2/D+3 snapshots
- Features: consumption estimates, weather, solar PV, feed-in, 
  calendar variables
- Result: **R² = 0.88, MAE = 0.0058 kWh**
- Outperforms D+2 baseline by **88%** and D+3 baseline by **35%**

### Stage 2 — XGBoost Anomaly Classifier
- Classifies prediction residuals (|D+2 − Predicted GT|) as 
  anomaly or normal
- Key innovation: engineered lag features at 15-min, 30-min, 
  1-hour, and 6-hour horizons give XGBoost memory of sequential 
  consumption history — addressing the core limitation of Random 
  Forest, which treats each timestamp independently
- Validated against EVN's rule-based plausibility benchmarks

---

## Key Design Decisions

| Decision | Reasoning |
|----------|-----------|
| Chronological train/test split | Prevents future data leakage |
| Plausibility flags excluded from features | Avoids circular logic |
| D+3 excluded from XGBoost features | Prevents leakage at D+2 decision point |
| Lag features added for XGBoost | RF misses sequential patterns |
| Residuals as anomaly signal | Richer semantic signal than raw values |

---

## Results

| Method | MAE (kWh) | Improvement over D+2 |
|--------|-----------|----------------------|
| D+2 baseline | 0.0452 | — |
| D+3 baseline | 0.0087 | 81% |
| Random Forest (Stage 1) | 0.0058 | 88% |

---

## Tech Stack

- **Language:** Python 3.9+
- **ML Models:** Scikit-learn (Random Forest), XGBoost
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Methodology:** CRISP-DM

---

## Project Structure
```
├── notebooks/          # Main Jupyter notebook (full pipeline)
├── src/                # Extracted Python modules
└── requirements.txt    # Dependencies
```

---

## Methodology: CRISP-DM

This project follows the full CRISP-DM lifecycle:
1. **Business Understanding** — EVN operational planning problem
2. **Data Understanding** — 5 data sources, 15-min smart meter data
3. **Data Preparation** — Merging, resampling, feature engineering
4. **Modeling** — Two-stage hybrid pipeline
5. **Evaluation** — MAE/RMSE/R², precision/recall, confusion matrix
6. **Anomaly Detection** — Residual-based XGBoost classifier

---



