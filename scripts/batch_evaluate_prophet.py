"""
batch_evaluate_prophet.py
-------------------------
Evaluasi otomatis model Prophet untuk setiap kombinasi (store, item)
Hasil: metrics.csv di folder reports/evaluation/
"""

from pathlib import Path
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# ======================
# CONFIG
# ======================
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "demand-forecasting-kernels-only" / "be-train.csv"
OUTDIR = ROOT / "reports" / "evaluation"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Jumlah hari terakhir untuk validasi
VAL_DAYS = 90


def evaluate_prophet(store_id: int, item_id: int, df: pd.DataFrame):
    """Latih Prophet untuk (store, item) dan hitung metrik performa."""
    subset = df[(df["store"] == store_id) & (df["item"] == item_id)].copy()
    subset = subset.sort_values("date")

    # Pisahkan train dan validation
    train = subset.iloc[:-VAL_DAYS]
    val = subset.iloc[-VAL_DAYS:]

    # Format untuk Prophet
    train = train.rename(columns={"date": "ds", "sales": "y"})

    # Model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode="additive",
        changepoint_prior_scale=0.5,
    )

    # Fit model
    model.fit(train)

    # Forecast periode validasi
    future = model.make_future_dataframe(periods=VAL_DAYS)
    forecast = model.predict(future)

    # Ambil hasil prediksi untuk validation range
    pred = forecast.set_index("ds").loc[val["date"], "yhat"].values
    y_true = val["sales"].values

    # Hitung metrik
    mae = mean_absolute_error(y_true, pred)
    rmse = np.sqrt(mean_squared_error(y_true, pred))
    mape = np.mean(np.abs((y_true - pred) / y_true)) * 100

    return mae, rmse, mape


def main():
    print("📦 Loading dataset...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(df.head())

    results = []

    stores = df["store"].unique()
    items = df["item"].unique()

    print(f"🔁 Evaluating {len(stores)} stores × {len(items)} items...")

    for s in stores:
        for i in items:
            try:
                mae, rmse, mape = evaluate_prophet(s, i, df)
                results.append({
                    "store": s,
                    "item": i,
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE (%)": mape
                })
                print(f"✅ Store {s} Item {i} done — MAPE: {mape:.2f}%")

            except Exception as e:
                print(f"❌ Store {s} Item {i} failed: {e}")

    # Simpan hasil
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTDIR / "summary_metrics.csv", index=False)

    print("\n✨ Evaluasi selesai!")
    print(f"📁 Hasil disimpan di: {OUTDIR / 'summary_metrics.csv'}")

    # Tampilkan top 5 dan worst 5
    print("\n🏆 Top 5 Akurasi (MAPE terendah):")
    print(results_df.sort_values("MAPE (%)").head(5))

    print("\n⚠️ Worst 5 Akurasi (MAPE tertinggi):")
    print(results_df.sort_values("MAPE (%)", ascending=False).head(5))


if __name__ == "__main__":
    main()
