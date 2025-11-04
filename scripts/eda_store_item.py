# scripts/eda_store_item.py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============
# CONFIG PATHS
# ============
ROOT = Path(__file__).resolve().parents[1]    # folder project
DATA_PATH = ROOT / "data" / "raw" / "demand-forecasting-kernels-only" / "be-train.csv"
OUTDIR = ROOT / "reports" / "eda"
OUTDIR.mkdir(parents=True, exist_ok=True)

def main():
    # 1) LOAD
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    # pastikan kolom sesuai
    expected_cols = {"date","store","item","sales"}
    assert expected_cols.issubset(df.columns), f"Kolom wajib hilang: {expected_cols - set(df.columns)}"

    # 2) INFO & DESCRIPTIVE STATS
    info = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.apply(lambda x: str(x)).to_dict(),
        "missing_per_col": df.isna().sum().to_dict(),
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
        "n_stores": df["store"].nunique(),
        "n_items": df["item"].nunique(),
        "sales_describe": df["sales"].describe().to_dict(),
    }
    pd.Series(info).to_json(OUTDIR / "summary_info.json", force_ascii=False)

    # 3) DUPLICATE CHECK
    # baris unik seharusnya = satu record per (date,store,item)
    dup_mask = df.duplicated(subset=["date","store","item"], keep=False)
    n_dup = int(dup_mask.sum())
    pd.Series({"n_duplicates": n_dup}).to_json(OUTDIR / "duplicates.json", force_ascii=False)
    if n_dup > 0:
        df[dup_mask].to_csv(OUTDIR / "duplicate_rows.csv", index=False)

    # 4) TIME INDEX CONSISTENCY (COVERAGE)
    # buat kalender lengkap harian untuk seluruh rentang
    full_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    expected_len = len(full_dates)

    # hitung jumlah tanggal unik per (store,item)
    coverage = (
        df.groupby(["store","item"])["date"].nunique()
          .rename("n_unique_dates")
          .reset_index()
    )
    coverage["expected_dates"] = expected_len
    coverage["is_complete_daily"] = coverage["n_unique_dates"].eq(expected_len)
    coverage.to_csv(OUTDIR / "coverage_per_series.csv", index=False)

    completeness_rate = coverage["is_complete_daily"].mean()
    pd.Series({"completeness_rate": float(completeness_rate)}).to_json(
        OUTDIR / "coverage_summary.json", force_ascii=False
    )

    # 5) AGGREGATIONS (EDA RINGKAS)
    # total penjualan harian (global)
    daily_sales = df.groupby("date")["sales"].sum()
    daily_sales.to_csv(OUTDIR / "daily_sales.csv", header=True)

    # rata-rata penjualan per store/item
    avg_sales_store = df.groupby("store")["sales"].mean().rename("avg_sales_store")
    avg_sales_item  = df.groupby("item")["sales"].mean().rename("avg_sales_item")
    avg_sales_store.to_csv(OUTDIR / "avg_sales_per_store.csv", header=True)
    avg_sales_item.to_csv(OUTDIR / "avg_sales_per_item.csv", header=True)

    # 6) QUICK QUALITY GUARDS
    # angka sales harus non-negatif dan integer
    invalid_sales = df.loc[(df["sales"] < 0) | (df["sales"] % 1 != 0)]
    pd.Series({"n_invalid_sales": len(invalid_sales)}).to_json(
        OUTDIR / "invalid_sales.json", force_ascii=False
    )
    if not invalid_sales.empty:
        invalid_sales.to_csv(OUTDIR / "invalid_sales_rows.csv", index=False)

    # cek tanggal out-of-range (harus berada antara min & max)
    date_min, date_max = df["date"].min(), df["date"].max()
    out_of_range = df[(df["date"] < date_min) | (df["date"] > date_max)]
    pd.Series({"n_out_of_range": len(out_of_range)}).to_json(
        OUTDIR / "date_out_of_range.json", force_ascii=False
    )

    # 7) VISUALISASI SEDERHANA (tanpa seaborn)
    plt.figure()
    daily_sales.plot(title="Total Daily Sales (All Stores & Items)")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.savefig(OUTDIR / "plot_daily_sales.png", dpi=150)

    # histogram sales (global)
    plt.figure()
    df["sales"].hist(bins=50)
    plt.title("Distribution of Daily Sales (per row)")
    plt.xlabel("sales")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(OUTDIR / "hist_sales.png", dpi=150)

    print("EDA Level 1 selesai. Cek folder:", OUTDIR)

if __name__ == "__main__":
    main()
