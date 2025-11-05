# scripts/eda_store_item_2.py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

# ============
# CONFIG
# ============
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "demand-forecasting-kernels-only" / "be-train.csv"
OUTDIR = ROOT / "reports" / "eda2"
OUTDIR.mkdir(parents=True, exist_ok=True)

# contoh seri yang akan di-zoom:
EXAMPLE_STORE = 1
EXAMPLE_ITEM  = 1

def main():
    # 1) LOAD
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values(["store","item","date"])
    assert {"date","store","item","sales"}.issubset(df.columns)

    # 2) ADD CALENDAR KEYS
    df["dow"] = df["date"].dt.dayofweek      # 0=Mon, 6=Sun
    df["dom"] = df["date"].dt.day            # 1..31
    df["moy"] = df["date"].dt.month          # 1..12
    df["year"] = df["date"].dt.year

    # 3) GLOBAL AGGREGATIONS (SEASONALITY)
    # DOW pattern (rata-rata penjualan harian)
    dow_mean = df.groupby("dow")["sales"].mean().rename("mean_sales_dow")
    dom_mean = df.groupby("dom")["sales"].mean().rename("mean_sales_dom")
    moy_mean = df.groupby("moy")["sales"].mean().rename("mean_sales_moy")

    dow_mean.to_csv(OUTDIR / "mean_sales_by_dow.csv", header=True)
    dom_mean.to_csv(OUTDIR / "mean_sales_by_dom.csv", header=True)
    moy_mean.to_csv(OUTDIR / "mean_sales_by_moy.csv", header=True)

    # Plot DOW
    plt.figure()
    dow_mean.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    dow_mean.plot(marker="o", title="Avg Sales by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Avg Sales")
    plt.tight_layout()
    plt.savefig(OUTDIR / "plot_avg_sales_by_dow.png", dpi=150)

    # Plot DOM
    plt.figure()
    dom_mean.plot(title="Avg Sales by Day of Month")
    plt.xlabel("Day of Month")
    plt.ylabel("Avg Sales")
    plt.tight_layout()
    plt.savefig(OUTDIR / "plot_avg_sales_by_dom.png", dpi=150)

    # Plot MOY
    plt.figure()
    moy_mean.index = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    moy_mean.plot(marker="o", title="Avg Sales by Month")
    plt.xlabel("Month")
    plt.ylabel("Avg Sales")
    plt.tight_layout()
    plt.savefig(OUTDIR / "plot_avg_sales_by_moy.png", dpi=150)

    # 4) PER-STORE / PER-ITEM VARIABILITY
    by_store = df.groupby("store")["sales"].agg(["mean","std"])
    by_store["cv"] = by_store["std"] / by_store["mean"]
    by_item  = df.groupby("item")["sales"].agg(["mean","std"])
    by_item["cv"] = by_item["std"] / by_item["mean"]

    by_store.to_csv(OUTDIR / "variability_by_store.csv")
    by_item.to_csv(OUTDIR / "variability_by_item.csv")

    # Plot CV store
    plt.figure()
    by_store["cv"].sort_values().plot(kind="bar", title="Coefficient of Variation by Store")
    plt.xlabel("Store")
    plt.ylabel("CV (std / mean)")
    plt.tight_layout()
    plt.savefig(OUTDIR / "plot_cv_by_store.png", dpi=150)

    # Plot CV item (Top 30)
    plt.figure()
    by_item["cv"].sort_values(ascending=False).head(30).plot(kind="bar", title="Top 30 Items by CV")
    plt.xlabel("Item")
    plt.ylabel("CV (std / mean)")
    plt.tight_layout()
    plt.savefig(OUTDIR / "plot_cv_top_items.png", dpi=150)

    # 5) ZOOM-IN: satu seri (store, item)
    one = df[(df["store"] == EXAMPLE_STORE) & (df["item"] == EXAMPLE_ITEM)].copy()
    one_daily = one.set_index("date")["sales"].asfreq("D")  # harus lengkap harian

    # Line plot
    plt.figure()
    one_daily.plot(title=f"Store {EXAMPLE_STORE} - Item {EXAMPLE_ITEM} Daily Sales")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"plot_series_store{EXAMPLE_STORE}_item{EXAMPLE_ITEM}.png", dpi=150)

    # DOW pattern untuk seri ini saja
    one["dow"] = one["date"].dt.dayofweek
    one_dow = one.groupby("dow")["sales"].mean()
    plt.figure()
    one_dow.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    one_dow.plot(marker="o", title=f"DOW Pattern — Store {EXAMPLE_STORE} Item {EXAMPLE_ITEM}")
    plt.xlabel("Day of Week")
    plt.ylabel("Avg Sales")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"plot_dow_store{EXAMPLE_STORE}_item{EXAMPLE_ITEM}.png", dpi=150)

    # 6) AUTOCORRELATION (ACF) — cek lag mingguan
    # Ambil 2 tahun terakhir biar komputasi ringan (opsional)
    one_recent = one_daily.dropna().iloc[-730:]
    acf_vals = acf(one_recent, nlags=30, fft=True)  # sampai lag 30 hari
    pd.Series(acf_vals).to_csv(OUTDIR / f"acf_store{EXAMPLE_STORE}_item{EXAMPLE_ITEM}.csv", header=False)

    plt.figure()
    pd.Series(acf_vals).plot(kind="bar", title=f"ACF (nlags=30) — Store {EXAMPLE_STORE} Item {EXAMPLE_ITEM}")
    plt.xlabel("Lag (days)")
    plt.ylabel("ACF")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"plot_acf_store{EXAMPLE_STORE}_item{EXAMPLE_ITEM}.png", dpi=150)

     # 7) SIMPAN RINGKASAN TEKS
    # cari hari (0–6) dengan rata-rata penjualan tertinggi
    global_dow_peak_num = int(df.groupby("dow")["sales"].mean().idxmax())
    # ubah jadi nama hari untuk ditampilkan
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    global_dow_peak_name = dow_names[global_dow_peak_num]

    summary = {
        "example_store": EXAMPLE_STORE,
        "example_item": EXAMPLE_ITEM,
        "date_min": df["date"].min().strftime("%Y-%m-%d"),
        "date_max": df["date"].max().strftime("%Y-%m-%d"),
        "n_stores": int(df["store"].nunique()),
        "n_items": int(df["item"].nunique()),
        "global_dow_peak": global_dow_peak_name,   # nama hari
        "global_moy_peak": int(df.groupby("moy")["sales"].mean().idxmax()),
    }

    pd.Series(summary).to_json(OUTDIR / "eda2_summary.json", force_ascii=False)


    print("EDA Level 2 selesai. Cek folder:", OUTDIR)

if __name__ == "__main__":
    main()
