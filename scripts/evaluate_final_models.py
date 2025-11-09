"""
evaluate_final_models.py
------------------------
Membandingkan Prophet, Hybrid, dan Hybrid Tuned XGBoost
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
HYBRID_METRICS = ROOT / "reports" / "hybrid_evaluation" / "hybrid_vs_prophet_metrics.csv"
TUNED_MODEL = ROOT / "models" / "hybrid_xgb_tuned.pkl"
HYBRID_TRAIN = ROOT / "data" / "processed" / "hybrid_train.csv"
OUTDIR = ROOT / "reports" / "final_evaluation"
OUTDIR.mkdir(parents=True, exist_ok=True)

def main():
    print("📦 Loading data and tuned model...")
    df = pd.read_csv(HYBRID_TRAIN, parse_dates=["date"])
    metrics = pd.read_csv(HYBRID_METRICS)
    model = joblib.load(TUNED_MODEL)

    print("🔍 Evaluating tuned hybrid model performance...")
    results = []
    for (store, item), group in df.groupby(["store", "item"]):
        group = group.sort_values("date")
        train = group.iloc[:-90]
        val = group.iloc[-90:]

        features = ["yhat", "dayofweek", "month", "year", "lag_1", "lag_7", "rolling_mean_7", "rolling_std_7"]

        # Pastikan kolom yang dibutuhkan ada
        if "rolling_std_7" not in val.columns:
            val["rolling_std_7"] = val.groupby(["store", "item"])["sales"].shift(1).rolling(7).std().reset_index(level=[0,1], drop=True)
        if "year" not in val.columns:
            val["year"] = val["date"].dt.year

        X_val = val[features]
        y_true = val["sales"].values
        yhat_prophet = val["yhat"].values
        yhat_tuned_resid = model.predict(X_val)

        # Hybrid Tuned prediction
        yhat_final = yhat_prophet + yhat_tuned_resid


        mae = mean_absolute_error(y_true, yhat_final)
        rmse = np.sqrt(mean_squared_error(y_true, yhat_final))
        mape = np.mean(np.abs((y_true - yhat_final) / y_true)) * 100

        results.append({
            "store": store,
            "item": item,
            "Hybrid_Tuned_MAE": mae,
            "Hybrid_Tuned_RMSE": rmse,
            "Hybrid_Tuned_MAPE": mape
        })

    result_df = pd.DataFrame(results)
    merged = metrics.merge(result_df, on=["store", "item"], how="left")
    merged.to_csv(OUTDIR / "final_model_comparison.csv", index=False)

    print(f"✅ Final comparison saved to: {OUTDIR / 'final_model_comparison.csv'}")
    print("📊 Sample preview:")
    print(merged.head())

if __name__ == "__main__":
    main()
