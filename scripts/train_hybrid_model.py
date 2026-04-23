"""
train_hybrid_model.py
---------------------
Melatih global XGBoost untuk residual correction pada hybrid forecasting.

Input:
- data/processed/hybrid_train.csv
- data/processed/hybrid_val.csv

Output:
- models/hybrid_xgb_model.pkl
- reports/hybrid_evaluation/hybrid_val_metrics.csv
- reports/hybrid_evaluation/hybrid_val_summary.json

Konsep:
1. Train XGBoost pada residual data train
2. Evaluasi di validation set
3. Bandingkan Prophet vs Hybrid pada validation set
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ======================
# PATH CONFIG
# ======================
ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "processed" / "hybrid_train.csv"
VAL_PATH = ROOT / "data" / "processed" / "hybrid_val.csv"

OUTDIR = ROOT / "reports" / "hybrid_evaluation"
MODELDIR = ROOT / "models"

OUTDIR.mkdir(parents=True, exist_ok=True)
MODELDIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELDIR / "hybrid_xgb_model.pkl"
METRICS_PATH = OUTDIR / "hybrid_val_metrics.csv"
SUMMARY_PATH = OUTDIR / "hybrid_val_summary.json"

# ======================
# FEATURE CONFIG
# ======================
FEATURES = [
    "yhat",
    "dayofweek",
    "month",
    "year",
    "dayofmonth",
    "is_weekend",
    "store_id",
    "item_id",
    "lag_yhat_1",
    "lag_yhat_7",
    "rolling_yhat_mean_7",
    "rolling_yhat_std_7",
    "lag_sales_1",
    "lag_sales_7",
    "rolling_sales_mean_7",
    "rolling_sales_std_7",
]

TARGET = "residual"


# ======================
# METRICS
# ======================
def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE aman untuk data yang bisa punya nilai 0."""
    denom = np.maximum(np.abs(y_true), 1)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """SMAPE lebih robust untuk demand forecasting."""
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1, denom)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = safe_mape(y_true, y_pred)
    s_mape = smape(y_true, y_pred)
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "SMAPE": float(s_mape),
    }


# ======================
# VALIDATION CHECKS
# ======================
def validate_columns(df: pd.DataFrame, required_cols: list[str], df_name: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def ensure_no_missing_features(df: pd.DataFrame, feature_cols: list[str], df_name: str) -> None:
    missing_count = df[feature_cols].isna().sum().sum()
    if missing_count > 0:
        raise ValueError(f"{df_name} still contains missing values in feature columns: {missing_count}")


# ======================
# MAIN
# ======================
def main():
    print("Loading train and validation datasets...")
    train_df = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    val_df = pd.read_csv(VAL_PATH, parse_dates=["date"])

    required_cols = ["date", "store", "item", "sales", "yhat", TARGET] + FEATURES
    validate_columns(train_df, required_cols, "hybrid_train.csv")
    validate_columns(val_df, required_cols, "hybrid_val.csv")

    ensure_no_missing_features(train_df, FEATURES + [TARGET], "hybrid_train.csv")
    ensure_no_missing_features(val_df, FEATURES, "hybrid_val.csv")

    print(f"Train rows: {len(train_df):,}")
    print(f"Val rows:   {len(val_df):,}")

    # ======================
    # TRAIN GLOBAL XGBOOST
    # ======================
    print("\nTraining global XGBoost residual model...")
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Save trained model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    # ======================
    # VALIDATION EVALUATION
    # ======================
    print("\nEvaluating on validation set...")
    X_val = val_df[FEATURES]
    y_true = val_df["sales"].values

    # Prophet baseline
    yhat_prophet = val_df["yhat"].values

    # Hybrid prediction
    residual_pred = model.predict(X_val)
    yhat_hybrid = yhat_prophet + residual_pred

    prophet_metrics_global = evaluate_predictions(y_true, yhat_prophet)
    hybrid_metrics_global = evaluate_predictions(y_true, yhat_hybrid)

    print("\nGlobal validation metrics:")
    print("Prophet:", prophet_metrics_global)
    print("Hybrid :", hybrid_metrics_global)

    # ======================
    # PER-SERIES METRICS
    # ======================
    print("\nComputing per-series validation metrics...")
    results = []

    grouped = val_df.copy()
    grouped["prophet_pred"] = yhat_prophet
    grouped["hybrid_pred"] = yhat_hybrid

    for (store, item), group in grouped.groupby(["store", "item"]):
        y = group["sales"].values
        prophet_pred = group["prophet_pred"].values
        hybrid_pred = group["hybrid_pred"].values

        prophet_m = evaluate_predictions(y, prophet_pred)
        hybrid_m = evaluate_predictions(y, hybrid_pred)

        results.append({
            "store": int(store),
            "item": int(item),

            "Prophet_MAE": prophet_m["MAE"],
            "Prophet_RMSE": prophet_m["RMSE"],
            "Prophet_MAPE": prophet_m["MAPE"],
            "Prophet_SMAPE": prophet_m["SMAPE"],

            "Hybrid_MAE": hybrid_m["MAE"],
            "Hybrid_RMSE": hybrid_m["RMSE"],
            "Hybrid_MAPE": hybrid_m["MAPE"],
            "Hybrid_SMAPE": hybrid_m["SMAPE"],

            "MAE_Improvement": prophet_m["MAE"] - hybrid_m["MAE"],
            "RMSE_Improvement": prophet_m["RMSE"] - hybrid_m["RMSE"],
            "MAPE_Improvement": prophet_m["MAPE"] - hybrid_m["MAPE"],
            "SMAPE_Improvement": prophet_m["SMAPE"] - hybrid_m["SMAPE"],
        })

    results_df = pd.DataFrame(results).sort_values(["store", "item"]).reset_index(drop=True)
    results_df.to_csv(METRICS_PATH, index=False)
    print(f"Per-series validation metrics saved to: {METRICS_PATH}")

    # ======================
    # SUMMARY
    # ======================
    improved_smape = int((results_df["SMAPE_Improvement"] > 0).sum())
    total_series = int(len(results_df))

    summary = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "n_series": total_series,

        "global_prophet_metrics": prophet_metrics_global,
        "global_hybrid_metrics": hybrid_metrics_global,

        "avg_prophet_mae": float(results_df["Prophet_MAE"].mean()),
        "avg_prophet_rmse": float(results_df["Prophet_RMSE"].mean()),
        "avg_prophet_mape": float(results_df["Prophet_MAPE"].mean()),
        "avg_prophet_smape": float(results_df["Prophet_SMAPE"].mean()),

        "avg_hybrid_mae": float(results_df["Hybrid_MAE"].mean()),
        "avg_hybrid_rmse": float(results_df["Hybrid_RMSE"].mean()),
        "avg_hybrid_mape": float(results_df["Hybrid_MAPE"].mean()),
        "avg_hybrid_smape": float(results_df["Hybrid_SMAPE"].mean()),

        "series_improved_on_smape": improved_smape,
        "series_improved_on_smape_pct": float((improved_smape / total_series) * 100.0),
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Validation summary saved to: {SUMMARY_PATH}")

    print("\nDone.")
    print(f"Series improved on SMAPE: {improved_smape}/{total_series} "
          f"({(improved_smape / total_series) * 100:.2f}%)")


if __name__ == "__main__":
    main()