"""
train_hybrid_model.py
---------------------
Latih model hybrid Prophet + XGBoost untuk meningkatkan akurasi forecast.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ======================
# PATH CONFIG
# ======================
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "hybrid_train.csv"
OUTDIR = ROOT / "reports" / "hybrid_evaluation"
MODELDIR = ROOT / "models"
OUTDIR.mkdir(parents=True, exist_ok=True)
MODELDIR.mkdir(parents=True, exist_ok=True)

VAL_DAYS = 90

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape

def main():
    print("📦 Loading hybrid dataset...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    features = ["dayofweek", "month", "lag_1", "lag_7", "rolling_mean_7"]
    
    stores = df["store"].unique()
    items = df["item"].unique()
    
    all_results = []
    for s in stores:
        for i in items:
            subset = df[(df["store"] == s) & (df["item"] == i)].dropna().sort_values("date")
            if len(subset) < VAL_DAYS + 10:
                continue

            train = subset.iloc[:-VAL_DAYS]
            val = subset.iloc[-VAL_DAYS:]
            
            X_train, y_train = train[features], train["residual"]
            X_val, y_val = val[features], val["residual"]
            
            model = XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            y_val_pred = model.predict(X_val)
            
            # Hybrid forecast = Prophet forecast + predicted residual
            val["hybrid_forecast"] = val["yhat"] + y_val_pred
            
            prophet_mae, prophet_rmse, prophet_mape = evaluate(val["sales"], val["yhat"])
            hybrid_mae, hybrid_rmse, hybrid_mape = evaluate(val["sales"], val["hybrid_forecast"])
            
            all_results.append({
                "store": s, "item": i,
                "Prophet_MAE": prophet_mae,
                "Prophet_RMSE": prophet_rmse,
                "Prophet_MAPE": prophet_mape,
                "Hybrid_MAE": hybrid_mae,
                "Hybrid_RMSE": hybrid_rmse,
                "Hybrid_MAPE": hybrid_mape,
            })
            
            print(f"🏪 Store {s}, Item {i} — Prophet MAPE: {prophet_mape:.2f}% | Hybrid MAPE: {hybrid_mape:.2f}%")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTDIR / "hybrid_vs_prophet_metrics.csv", index=False)
    print(f"\n📊 Evaluation results saved to: {OUTDIR / 'hybrid_vs_prophet_metrics.csv'}")

    # Simpan model terakhir (optional)
    joblib.dump(model, MODELDIR / "hybrid_xgb_model.pkl")
    print(f"💾 XGBoost model saved to: {MODELDIR / 'hybrid_xgb_model.pkl'}")

if __name__ == "__main__":
    main()
