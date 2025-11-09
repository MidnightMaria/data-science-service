import pandas as pd
import numpy as np
import os

# === Configuration ===
Z = 1.65  # Z-score for 95% service level
ORDERING_COST = 50  # biaya pesan per order
HOLDING_COST = 2    # biaya simpan per unit per periode
LEAD_TIME_DAYS = 7  # rata-rata lead time (hari)
DAYS_IN_YEAR = 365

# === Paths ===
input_path = "reports/forecast/future_demand_forecast.csv"
output_dir = "reports/optimization"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "inventory_optimization_report.csv")

print("📦 Loading forecast data...")
df = pd.read_csv(input_path)

# Pastikan kolom sesuai format yang kita hasilkan dari forecasting
# Biasanya: store, item, ds, hybrid_forecast
if "hybrid_forecast" not in df.columns:
    raise ValueError("❌ Column 'hybrid_forecast' not found in forecast data!")

# === Aggregate demand forecast ===
agg = (
    df.groupby(["store", "item"])["hybrid_forecast"]
    .agg(["mean", "std", "sum"])
    .reset_index()
    .rename(columns={"mean": "mean_demand", "std": "std_demand", "sum": "total_demand"})
)

# === Compute Safety Stock, ROP, EOQ ===
print("⚙️ Calculating Safety Stock, ROP, and EOQ...")

agg["safety_stock"] = Z * agg["std_demand"] * np.sqrt(LEAD_TIME_DAYS)
agg["reorder_point"] = (agg["mean_demand"] * LEAD_TIME_DAYS) + agg["safety_stock"]

# Hitung EOQ berdasarkan demand tahunan (approx dari total_demand)
agg["annual_demand"] = agg["mean_demand"] * DAYS_IN_YEAR / len(df["ds"].unique())
agg["eoq"] = np.sqrt((2 * agg["annual_demand"] * ORDERING_COST) / HOLDING_COST)

# === Determine optimal stock level ===
agg["optimal_stock_level"] = agg["reorder_point"] + agg["eoq"]

# === Round & clean ===
agg = agg.round(2)

# === Save result ===
agg.to_csv(output_path, index=False)
print(f"✅ Inventory optimization report saved to: {output_path}")

print("\n📊 Sample preview:")
print(agg.head(10))
