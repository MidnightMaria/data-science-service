"""
evaluate_final_models.py
-----------------------
Final evaluation di TEST SET (NO LEAKAGE)

Input:
- hybrid_test.csv
- hybrid_xgb_tuned.pkl

Output:
- test_metrics.csv
- test_summary.json
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

OUTDIR = ROOT / "reports/final_evaluation"
OUTDIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = OUTDIR / "test_metrics.csv"
SUMMARY_PATH = OUTDIR / "test_summary.json"

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
# MAIN
# ======================
def main():
    print("Loading test data...")
    df = pd.read_csv(TEST_PATH, parse_dates=["date"])

    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)

    X = df[FEATURES]
    y_true = df["sales"].values

    # Prophet baseline
    yhat_prophet = df["yhat"].values

    # Hybrid prediction
    residual_pred = model.predict(X)
    yhat_hybrid = yhat_prophet + residual_pred

    # ======================
    # GLOBAL METRICS
    # ======================
    prophet_metrics = evaluate(y_true, yhat_prophet)
    hybrid_metrics = evaluate(y_true, yhat_hybrid)

    print("\nFINAL TEST RESULTS")
    print("Prophet:", prophet_metrics)
    print("Hybrid :", hybrid_metrics)

    # ======================
    # PER SERIES
    # ======================
    df["prophet_pred"] = yhat_prophet
    df["hybrid_pred"] = yhat_hybrid

    results = []

    for (store, item), group in df.groupby(["store", "item"]):
        y = group["sales"].values
        p = group["prophet_pred"].values
        h = group["hybrid_pred"].values

        pm = evaluate(y, p)
        hm = evaluate(y, h)

        results.append({
            "store": int(store),
            "item": int(item),

            "Prophet_SMAPE": pm["SMAPE"],
            "Hybrid_SMAPE": hm["SMAPE"],
            "SMAPE_Improvement": pm["SMAPE"] - hm["SMAPE"]
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(METRICS_PATH, index=False)

    improved = (results_df["SMAPE_Improvement"] > 0).sum()
    total = len(results_df)

    summary = {
        "global_prophet": prophet_metrics,
        "global_hybrid": hybrid_metrics,
        "series_improved": int(improved),
        "total_series": int(total),
        "improvement_pct": float(improved / total * 100)
    }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSeries improved:", improved, "/", total,
          f"({improved/total*100:.2f}%)")

    print("\nDone.")
    print("Saved to:", OUTDIR)

if __name__ == "__main__":
    main()