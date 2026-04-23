"""
plot_final_evaluation.py
-----------------------
Visualisasi hasil forecasting untuk laporan
"""

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# PATH
# ======================
ROOT = Path(__file__).resolve().parents[1]

BASELINE_PATH = ROOT / "reports/baseline_comparison/baseline_summary.json"
FINAL_PATH = ROOT / "reports/final_evaluation/test_metrics.csv"

OUTDIR = ROOT / "reports/plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ======================
# LOAD DATA
# ======================
with open(BASELINE_PATH) as f:
    baseline = json.load(f)

df = pd.read_csv(FINAL_PATH)

# ======================
# 1. BAR CHART (SMAPE)
# ======================
models = list(baseline.keys())
smape_values = [baseline[m]["SMAPE"] for m in models]

plt.figure()
plt.bar(models, smape_values)
plt.title("Model Comparison (SMAPE)")
plt.ylabel("SMAPE")
plt.savefig(OUTDIR / "smape_comparison.png")
plt.close()

# ======================
# 2. HISTOGRAM IMPROVEMENT
# ======================
df["SMAPE_Improvement"] = df["Prophet_SMAPE"] - df["Hybrid_SMAPE"]

plt.figure()
plt.hist(df["SMAPE_Improvement"], bins=30)
plt.title("SMAPE Improvement Distribution")
plt.xlabel("Improvement")
plt.savefig(OUTDIR / "improvement_hist.png")
plt.close()

# ======================
# 3. SCATTER PLOT
# ======================
plt.figure()
plt.scatter(df["Prophet_SMAPE"], df["Hybrid_SMAPE"])
plt.xlabel("Prophet SMAPE")
plt.ylabel("Hybrid SMAPE")
plt.title("Prophet vs Hybrid")

# diagonal line
plt.plot([0, 25], [0, 25])

plt.savefig(OUTDIR / "scatter.png")
plt.close()

# ======================
# 4. TOP & WORST SERIES
# ======================
best = df.sort_values("SMAPE_Improvement", ascending=False).head(10)
worst = df.sort_values("SMAPE_Improvement").head(10)

best.to_csv(OUTDIR / "best_series.csv", index=False)
worst.to_csv(OUTDIR / "worst_series.csv", index=False)

print("Plots saved to:", OUTDIR)