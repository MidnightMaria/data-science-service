"""
tune_hybrid_xgb.py
------------------
Hyperparameter tuning untuk XGBoost pada hybrid Prophet + ML model.
Menggunakan Grid Search dengan Cross Validation.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
import joblib
from pathlib import Path

# ========= CONFIG =========
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "hybrid_train.csv"
OUTDIR = ROOT / "reports" / "tuned_xgb"
MODEL_DIR = ROOT / "models"
OUTDIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ========= LOAD DATA =========
print("📦 Loading hybrid training dataset...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
print(f"✅ Loaded {len(df):,} rows.")

# ========= PREPARE FEATURES =========
features = ["yhat", "dayofweek", "month", "year", "lag_1", "lag_7", "rolling_mean_7", "rolling_std_7"]
target = "residual"

if target not in df.columns:
    df["residual"] = df["sales"] - df["yhat"]

X = df[features]

y = df[target]

# ========= TIME SERIES SPLIT =========
tscv = TimeSeriesSplit(n_splits=3)

# ========= GRID SEARCH =========
param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators": [100, 200],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.7, 1.0],
    "gamma": [0, 0.2, 0.5],
    "reg_lambda": [1, 2, 5],
}

xgb_model = XGBRegressor(objective="reg:squarederror", random_state=42)

print("🔍 Starting Grid Search for XGBoost...")
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X, y)
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print("\n🏆 Best Parameters Found:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# ========= EVALUATE BEST MODEL =========
y_pred = best_model.predict(X)
mape = mean_absolute_percentage_error(y, y_pred) * 100
print(f"\n📈 Training MAPE (residual prediction): {mape:.2f}%")

# ========= SAVE RESULTS =========
joblib.dump(best_model, MODEL_DIR / "hybrid_xgb_tuned.pkl")
pd.DataFrame([best_params]).to_csv(OUTDIR / "best_xgb_params.csv", index=False)
print(f"💾 Tuned model saved to: {MODEL_DIR / 'hybrid_xgb_tuned.pkl'}")
print(f"📊 Best params saved to: {OUTDIR / 'best_xgb_params.csv'}")

print("\n✨ XGBoost tuning complete!")
