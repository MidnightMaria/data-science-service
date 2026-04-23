"""
optimize_inventory.py
---------------------
Inventory optimization berbasis hasil forecast hybrid.

Input:
- data/processed/hybrid_test.csv
- models/hybrid_xgb_tuned.pkl

Output:
- reports/inventory_optimization/inventory_policy_report.csv
- reports/inventory_optimization/inventory_summary.json

Konsep:
1. Gunakan hybrid prediction untuk estimasi demand
2. Gunakan forecast error sebagai pendekatan uncertainty
3. Hitung Safety Stock, ROP, dan EOQ
4. Simulasikan current stock secara lebih realistis
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# ======================
# CONFIG
# ======================
ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = ROOT / "data" / "processed" / "hybrid_test.csv"
MODEL_PATH = ROOT / "models" / "hybrid_xgb_tuned.pkl"

OUTDIR = ROOT / "reports" / "inventory_optimization"
OUTDIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTDIR / "inventory_policy_report.csv"
SUMMARY_PATH = OUTDIR / "inventory_summary.json"

# Inventory assumptions
LEAD_TIME_DAYS = 7
SERVICE_LEVEL_Z = 1.65   # 95%
ORDERING_COST = 50.0
HOLDING_COST = 2.0
DAYS_IN_YEAR = 365

# Simulasi stok realistis
# artinya current stock diasumsikan setara dengan 3–14 hari demand rata-rata
STOCK_COVERAGE_MIN_DAYS = 3
STOCK_COVERAGE_MAX_DAYS = 14
RANDOM_SEED = 42

FEATURES = [
    "yhat", "dayofweek", "month", "year", "dayofmonth", "is_weekend",
    "store_id", "item_id",
    "lag_yhat_1", "lag_yhat_7",
    "rolling_yhat_mean_7", "rolling_yhat_std_7",
    "lag_sales_1", "lag_sales_7",
    "rolling_sales_mean_7", "rolling_sales_std_7"
]


def main():
    print("Loading hybrid test data...")
    df = pd.read_csv(TEST_PATH, parse_dates=["date"])

    print("Loading tuned hybrid model...")
    model = joblib.load(MODEL_PATH)

    rng = np.random.default_rng(RANDOM_SEED)

    # ======================
    # BUILD HYBRID PREDICTION
    # ======================
    X = df[FEATURES]
    df["residual_pred"] = model.predict(X)
    df["hybrid_pred"] = df["yhat"] + df["residual_pred"]

    # Forecast demand tidak boleh negatif
    df["hybrid_pred"] = df["hybrid_pred"].clip(lower=0)

    # Forecast error
    df["forecast_error"] = df["sales"] - df["hybrid_pred"]

    # ======================
    # AGGREGATE PER SERIES
    # ======================
    results = []

    for (store, item), group in df.groupby(["store", "item"]):
        group = group.sort_values("date").copy()

        actual = group["sales"].values
        pred = group["hybrid_pred"].values

        # Mean daily demand dari hybrid forecast
        mean_daily_demand = float(np.mean(pred))

        # RMSE sebagai pendekatan uncertainty
        rmse = float(np.sqrt(mean_squared_error(actual, pred)))

        # Safety Stock
        safety_stock = SERVICE_LEVEL_Z * rmse * np.sqrt(LEAD_TIME_DAYS)

        # Reorder Point
        reorder_point = (mean_daily_demand * LEAD_TIME_DAYS) + safety_stock

        # Annual demand untuk EOQ
        annual_demand = mean_daily_demand * DAYS_IN_YEAR

        # EOQ
        if HOLDING_COST > 0:
            eoq = float(np.sqrt((2 * annual_demand * ORDERING_COST) / HOLDING_COST))
        else:
            eoq = np.nan

        # ======================
        # REALISTIC CURRENT STOCK SIMULATION
        # ======================
        # Stok disimulasikan sebagai coverage 3–14 hari demand
        coverage_days = float(rng.uniform(STOCK_COVERAGE_MIN_DAYS, STOCK_COVERAGE_MAX_DAYS))
        current_stock_proxy = float(mean_daily_demand * coverage_days)

        # Inventory status
        if current_stock_proxy <= reorder_point:
            inventory_status = "REORDER"
        else:
            inventory_status = "SAFE"

        # Opsional: beri indikasi berapa banyak perlu order
        reorder_qty = max(eoq, reorder_point - current_stock_proxy) if inventory_status == "REORDER" else 0.0

        results.append({
            "store": int(store),
            "item": int(item),

            "mean_daily_demand": round(mean_daily_demand, 4),
            "rmse_forecast": round(rmse, 4),

            "lead_time_days": LEAD_TIME_DAYS,
            "service_level_z": SERVICE_LEVEL_Z,

            "safety_stock": round(float(safety_stock), 4),
            "reorder_point": round(float(reorder_point), 4),
            "annual_demand": round(float(annual_demand), 4),
            "eoq": round(float(eoq), 4),

            "stock_coverage_days": round(coverage_days, 4),
            "current_stock_proxy": round(current_stock_proxy, 4),
            "inventory_status": inventory_status,
            "recommended_order_qty": round(float(reorder_qty), 4)
        })

    result_df = pd.DataFrame(results).sort_values(["store", "item"]).reset_index(drop=True)
    result_df.to_csv(REPORT_PATH, index=False)

    # ======================
    # SUMMARY
    # ======================
    n_reorder = int((result_df["inventory_status"] == "REORDER").sum())
    n_safe = int((result_df["inventory_status"] == "SAFE").sum())

    summary = {
        "n_series": int(len(result_df)),
        "lead_time_days": LEAD_TIME_DAYS,
        "service_level_z": SERVICE_LEVEL_Z,
        "stock_coverage_min_days": STOCK_COVERAGE_MIN_DAYS,
        "stock_coverage_max_days": STOCK_COVERAGE_MAX_DAYS,
        "avg_mean_daily_demand": float(result_df["mean_daily_demand"].mean()),
        "avg_rmse_forecast": float(result_df["rmse_forecast"].mean()),
        "avg_safety_stock": float(result_df["safety_stock"].mean()),
        "avg_reorder_point": float(result_df["reorder_point"].mean()),
        "avg_eoq": float(result_df["eoq"].mean()),
        "avg_current_stock_proxy": float(result_df["current_stock_proxy"].mean()),
        "n_reorder": n_reorder,
        "n_safe": n_safe,
        "pct_reorder": float((n_reorder / len(result_df)) * 100.0),
        "pct_safe": float((n_safe / len(result_df)) * 100.0),
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Inventory policy report saved to: {REPORT_PATH}")
    print(f"Inventory summary saved to: {SUMMARY_PATH}")
    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()