"""
visualize_hybrid_results.py
----------------------------
Visualisasi perbandingan Prophet vs Hybrid model performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ========= CONFIG =========
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "reports" / "hybrid_evaluation" / "hybrid_vs_prophet_metrics.csv"
OUTDIR = ROOT / "reports" / "hybrid_evaluation" / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ========= LOAD =========
df = pd.read_csv(DATA_PATH)
print(f"📦 Loaded metrics data: {df.shape[0]} rows")

# ========= 1️⃣ Scatter Plot Prophet vs Hybrid (MAPE) =========
plt.figure(figsize=(7,7))
plt.scatter(df["Prophet_MAPE"], df["Hybrid_MAPE"], color="teal", alpha=0.7)
plt.plot([0, max(df["Prophet_MAPE"])], [0, max(df["Prophet_MAPE"])], "--", color="gray")
plt.title("Prophet vs Hybrid MAPE (%)")
plt.xlabel("Prophet MAPE (%)")
plt.ylabel("Hybrid MAPE (%)")
plt.tight_layout()
plt.savefig(OUTDIR / "scatter_prophet_vs_hybrid_mape.png", dpi=150)
print("✅ Scatter plot saved.")

# ========= 2️⃣ Bar Chart: Average MAPE per Store =========
avg_per_store = (
    df.groupby("store")[["Prophet_MAPE", "Hybrid_MAPE"]].mean().reset_index()
)

plt.figure(figsize=(8,5))
bar_width = 0.35
x = avg_per_store["store"]
plt.bar(x - bar_width/2, avg_per_store["Prophet_MAPE"], width=bar_width, label="Prophet", color="#f2a6a1")
plt.bar(x + bar_width/2, avg_per_store["Hybrid_MAPE"], width=bar_width, label="Hybrid", color="#a1d8f2")
plt.title("Average MAPE per Store")
plt.xlabel("Store ID")
plt.ylabel("Average MAPE (%)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "bar_avg_mape_per_store.png", dpi=150)
print("✅ Bar chart saved.")

# ========= 3️⃣ Delta Improvement Histogram =========
df["delta_mape"] = df["Prophet_MAPE"] - df["Hybrid_MAPE"]

plt.figure(figsize=(7,5))
plt.hist(df["delta_mape"], bins=10, color="#ffb6c1", edgecolor="black")
plt.axvline(0, color="gray", linestyle="--")
plt.title("Hybrid Improvement (Prophet MAPE - Hybrid MAPE)")
plt.xlabel("MAPE Improvement (%)")
plt.ylabel("Number of Store-Item pairs")
plt.tight_layout()
plt.savefig(OUTDIR / "hist_mape_improvement.png", dpi=150)
print("✅ Histogram saved.")

# ========= SAVE SUMMARY =========
improved = (df["delta_mape"] > 0).mean() * 100
worse = (df["delta_mape"] <= 0).mean() * 100
summary = {
    "total_items": len(df),
    "improved_%": improved,
    "worse_%": worse,
    "avg_prophet_mape": df["Prophet_MAPE"].mean(),
    "avg_hybrid_mape": df["Hybrid_MAPE"].mean(),
}
pd.Series(summary).to_json(OUTDIR / "hybrid_visual_summary.json", indent=2)
print("📊 Summary saved:", summary)

print(f"\n✨ Visualization complete! Check {OUTDIR}")
