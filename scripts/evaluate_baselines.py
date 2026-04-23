"""
evaluate_baselines.py
---------------------
Bandingkan baseline vs Prophet vs Hybrid di TEST SET
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ======================
# PATH
# ======================
ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = ROOT / "data/processed/hybrid_test.csv"
MODEL_PATH = ROOT / "models/hybrid_xgb_tuned.pkl"

OUTDIR = ROOT / "reports/baseline_comparison"
OUTDIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUTDIR / "baseline_summary.json"

# ======================
# FEATURES
# ======================
FEATURES = [
    "yhat","dayofweek","month","year","dayofmonth","is_weekend",
    "store_id","item_id",
    "lag_yhat_1","lag_yhat_7",
    "rolling_yhat_mean_7","rolling_yhat_std_7",
    "lag_sales_1","lag_sales_7",
    "rolling_sales_mean_7","rolling_sales_std_7"
]

# ======================
# METRICS
# ======================
def safe_mape(y_true, y_pred):
    denom = np.maximum(np.abs(y_true), 1)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

def smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1, denom)
    return np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100

def evaluate(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(safe_mape(y_true, y_pred)),
        "SMAPE": float(smape(y_true, y_pred)),
    }

# ======================
# BASELINE FUNCTIONS
# ======================
def naive_forecast(df):
    # pakai lag 1 (hari sebelumnya)
    return df["lag_sales_1"].values

def moving_average_forecast(df):
    # pakai rolling mean 7 hari
    return df["rolling_sales_mean_7"].values

# ======================
# MAIN
# ======================
def main():
    print("Loading test data...")
    df = pd.read_csv(TEST_PATH)

    print("Loading hybrid model...")
    model = joblib.load(MODEL_PATH)

    y_true = df["sales"].values

    # ======================
    # BASELINES
    # ======================
    print("\nEvaluating baselines...")

    naive_pred = naive_forecast(df)
    ma_pred = moving_average_forecast(df)

    # Prophet
    prophet_pred = df["yhat"].values

    # Hybrid
    X = df[FEATURES]
    residual_pred = model.predict(X)
    hybrid_pred = prophet_pred + residual_pred

    # ======================
    # METRICS
    # ======================
    results = {
        "Naive": evaluate(y_true, naive_pred),
        "Moving_Average": evaluate(y_true, ma_pred),
        "Prophet": evaluate(y_true, prophet_pred),
        "Hybrid": evaluate(y_true, hybrid_pred),
    }

    print("\nFINAL COMPARISON:")
    for name, metrics in results.items():
        print(name, ":", metrics)

    # ======================
    # SAVE
    # ======================
    with open(SUMMARY_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to:", SUMMARY_PATH)

if __name__ == "__main__":
    main()