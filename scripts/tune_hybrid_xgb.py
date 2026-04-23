"""
tune_hybrid_xgb.py
-----------------
Hyperparameter tuning XGBoost untuk hybrid model (NO LEAKAGE)

Input:
- hybrid_train.csv
- hybrid_val.csv

Output:
- best params
- best model disimpan
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from itertools import product

# ======================
# PATH
# ======================
ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data/processed/hybrid_train.csv"
VAL_PATH   = ROOT / "data/processed/hybrid_val.csv"

MODEL_PATH = ROOT / "models/hybrid_xgb_tuned.pkl"
RESULT_PATH = ROOT / "reports/hybrid_evaluation/tuning_results.json"

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

TARGET = "residual"

# ======================
# LOAD DATA
# ======================
train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_val = val_df[FEATURES]
y_val_actual = val_df["sales"].values
y_val_prophet = val_df["yhat"].values

# ======================
# PARAM GRID
# ======================
param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [4, 5, 6],
    "learning_rate": [0.03, 0.05],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}

keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in product(*values)]

print(f"Total combinations: {len(param_combinations)}")

# ======================
# TUNING LOOP
# ======================
best_score = float("inf")
best_params = None
best_model = None

results = []

for i, params in enumerate(param_combinations):
    print(f"\n[{i+1}/{len(param_combinations)}] Testing params: {params}")

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        **params
    )

    model.fit(X_train, y_train)

    # Predict residual
    residual_pred = model.predict(X_val)

    # Hybrid prediction
    y_pred = y_val_prophet + residual_pred

    # Use MAE for tuning (stable)
    mae = mean_absolute_error(y_val_actual, y_pred)

    print(f"MAE: {mae:.4f}")

    results.append({
        "params": params,
        "mae": float(mae)
    })

    if mae < best_score:
        best_score = mae
        best_params = params
        best_model = model

# ======================
# SAVE RESULTS
# ======================
print("\nBest Params:", best_params)
print("Best MAE:", best_score)

joblib.dump(best_model, MODEL_PATH)

with open(RESULT_PATH, "w") as f:
    json.dump({
        "best_params": best_params,
        "best_mae": best_score,
        "all_results": results
    }, f, indent=2)

print("\nTuning selesai.")
print(f"Best model saved to: {MODEL_PATH}")