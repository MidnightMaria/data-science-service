"""
plot_final_evaluation.py
------------------------
Visualisasi perbandingan Prophet vs Hybrid vs Hybrid Tuned
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "reports" / "final_evaluation" / "final_model_comparison.csv"
OUTDIR = ROOT / "reports" / "final_evaluation"
OUTDIR.mkdir(parents=True, exist_ok=True)

def main():
    print("📊 Loading evaluation results...")
    df = pd.read_csv(EVAL_PATH)

    # Hitung rata-rata MAPE tiap model
    avg_prophet = df["Prophet_MAPE"].mean()
    avg_hybrid = df["Hybrid_MAPE"].mean()
    avg_tuned = df["Hybrid_Tuned_MAPE"].mean()

    print(f"📈 Prophet avg MAPE: {avg_prophet:.2f}%")
    print(f"🔧 Hybrid avg MAPE: {avg_hybrid:.2f}%")
    print(f"🚀 Tuned Hybrid avg MAPE: {avg_tuned:.2f}%")

    # Hitung improvement per item
    df["Improvement_Hybrid_vs_Prophet"] = df["Prophet_MAPE"] - df["Hybrid_MAPE"]
    df["Improvement_Tuned_vs_Hybrid"] = df["Hybrid_MAPE"] - df["Hybrid_Tuned_MAPE"]

    improved_count = (df["Improvement_Tuned_vs_Hybrid"] > 0).sum()
    total = len(df)
    improved_pct = (improved_count / total) * 100

    print(f"✨ {improved_count}/{total} items improved ({improved_pct:.2f}%) after tuning.")

    # 1️⃣ Bar chart rata-rata MAPE per model
    plt.figure(figsize=(6, 4))
    plt.bar(["Prophet", "Hybrid", "Hybrid Tuned"], [avg_prophet, avg_hybrid, avg_tuned], color=["#e6a5a1", "#a7c7e7", "#f3d1a9"])
    plt.title("Average MAPE Comparison")
    plt.ylabel("MAPE (%)")
    plt.tight_layout()
    plt.savefig(OUTDIR / "avg_mape_comparison.png", dpi=150)
    plt.close()

    # 2️⃣ Line chart tren per model (contoh untuk 10 item pertama)
    plt.figure(figsize=(10, 5))
    subset = df.head(10)
    plt.plot(subset["item"], subset["Prophet_MAPE"], marker="o", label="Prophet", color="#e6a5a1")
    plt.plot(subset["item"], subset["Hybrid_MAPE"], marker="o", label="Hybrid", color="#a7c7e7")
    plt.plot(subset["item"], subset["Hybrid_Tuned_MAPE"], marker="o", label="Hybrid Tuned", color="#f3d1a9")
    plt.title("MAPE per Item (Top 10)")
    plt.xlabel("Item")
    plt.ylabel("MAPE (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "mape_per_item_trend.png", dpi=150)
    plt.close()

    # 3️⃣ Histogram improvement
    plt.figure(figsize=(6,4))
    plt.hist(df["Improvement_Tuned_vs_Hybrid"], bins=15, color="#f3d1a9", edgecolor="black")
    plt.title("Distribution of Improvement (Tuned vs Hybrid)")
    plt.xlabel("MAPE Improvement (%)")
    plt.ylabel("Number of Items")
    plt.tight_layout()
    plt.savefig(OUTDIR / "improvement_distribution.png", dpi=150)
    plt.close()

    print(f"✅ Visualizations saved to: {OUTDIR}")

if __name__ == "__main__":
    main()
