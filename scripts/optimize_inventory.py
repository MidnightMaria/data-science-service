import pandas as pd
import numpy as np
import os

# Configuration

Z = 1.65            # Z-score for 95% service level
ORDERING_COST = 50  # ordering cost per order
HOLDING_COST = 2    # holding cost per unit per year
LEAD_TIME_DAYS = 7  # lead time in days
DAYS_IN_YEAR = 365

# Paths

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_path = os.path.join(BASE_DIR, "reports", "forecast", "future_demand_forecast.csv")
output_dir = os.path.join(BASE_DIR, "reports", "optimization")

os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "inventory_optimization_report.csv")

print("Loading forecast data...")
df = pd.read_csv(input_path)


# Validate Columns


if "hybrid_forecast" not in df.columns:
    raise ValueError("Column 'hybrid_forecast' not found in forecast data")


# Aggregate Demand Forecast


print("Aggregating demand...")

agg = (
    df.groupby(["store", "item"])["hybrid_forecast"]
    .agg(["mean", "std", "sum"])
    .reset_index()
    .rename(columns={
        "mean": "mean_demand",
        "std": "std_demand",
        "sum": "total_demand"
    })
)


# Inventory Calculations


print("Calculating Safety Stock, ROP, EOQ...")

# Safety Stock
agg["safety_stock"] = Z * agg["std_demand"] * np.sqrt(LEAD_TIME_DAYS)

# Reorder Point
agg["reorder_point"] = (agg["mean_demand"] * LEAD_TIME_DAYS) + agg["safety_stock"]

# Annual Demand
agg["annual_demand"] = agg["mean_demand"] * DAYS_IN_YEAR

# EOQ
agg["eoq"] = np.sqrt((2 * agg["annual_demand"] * ORDERING_COST) / HOLDING_COST)

# Optimal Stock Level
agg["optimal_stock_level"] = agg["reorder_point"] + agg["eoq"]


# Decision Support Logic


print("Generating decision support recommendations...")


# Simulated Current Stock


np.random.seed(42)

stock_min = (agg["reorder_point"] * 0.5).astype(int)
stock_max = (agg["optimal_stock_level"] * 1.2).astype(int)

agg["current_stock"] = [
    np.random.randint(low, high)
    for low, high in zip(stock_min, stock_max)
]

# Status determination
agg["status"] = np.where(
    agg["current_stock"] <= agg["reorder_point"],
    "ORDER NOW",
    np.where(
        agg["current_stock"] > agg["optimal_stock_level"],
        "OVERSTOCK",
        "SAFE"
    )
)

agg["risk_level"] = np.where(
    agg["current_stock"] < agg["reorder_point"] * 0.7,
    "CRITICAL",
    np.where(
        agg["current_stock"] <= agg["reorder_point"],
        "LOW STOCK",
        "NORMAL"
    )
)

# Recommended order quantity
agg["recommended_order_qty"] = np.where(
    agg["status"] == "ORDER NOW",
    np.maximum(agg["eoq"], agg["optimal_stock_level"] - agg["current_stock"]),
    0
)

# Cost Estimation

agg["holding_cost_est"] = agg["current_stock"] * HOLDING_COST
agg["ordering_cost_est"] = np.where(
    agg["status"] == "ORDER NOW",
    ORDERING_COST,
    0
)

agg["total_cost_est"] = agg["holding_cost_est"] + agg["ordering_cost_est"]

# Cleanup and Save

agg = agg.round(2)

agg.to_csv(output_path, index=False)

print(f"Inventory optimization report saved to: {output_path}")

print("Sample preview:")
print(agg.head(10))