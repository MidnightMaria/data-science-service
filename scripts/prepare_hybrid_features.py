"""
prepare_hybrid_features.py
--------------------------
Mempersiapkan dataset gabungan untuk hybrid forecasting:
Prophet (trend + seasonality) + XGBoost (residual correction)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from prophet import Prophet

# =======================
# PATH CONFIG
# =======================
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "demand-forecasting-kernels-only" / "be-train.csv"
OUTDIR = ROOT / "data" / "processed"
OUTDIR.mkdir(parents=True, exist_ok=True)

VAL_DAYS = 90

# =======================
# HELPER FUNCTIONS
# =======================
def generate_prophet_forecast(df, store, item):
    """Fit Prophet per kombinasi store-item dan hasilkan yhat (forecast)."""
    subset = df[(df["store"] == store) & (df["item"] == item)].copy().sort_values("date")
    train = subset.iloc[:-VAL_DAYS].rename(columns={"date": "ds", "sales": "y"})

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(train)

    future = model.make_future_dataframe(periods=VAL_DAYS)
    forecast = model.predict(future)
    merged = subset.merge(forecast[["ds", "yhat"]], left_on="date", right_on="ds", how="left")

    merged["residual"] = merged["sales"] - merged["yhat"]
    merged["store"] = store
    merged["item"] = item
    merged.drop(columns=["ds"], inplace=True)
    return merged


def add_time_features(df):
    """Tambah semua fitur waktu, lag, rolling mean, rolling std."""
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    # fitur lag dan rolling berdasarkan kombinasi store-item
    df["lag_1"] = df.groupby(["store", "item"])["sales"].shift(1)
    df["lag_7"] = df.groupby(["store", "item"])["sales"].shift(7)
    df["rolling_mean_7"] = df.groupby(["store", "item"])["sales"].transform(lambda x: x.shift(1).rolling(7).mean())
    df["rolling_std_7"]  = df.groupby(["store", "item"])["sales"].transform(lambda x: x.shift(1).rolling(7).std())

    df = df.dropna().reset_index(drop=True)
    return df


# =======================
# MAIN PIPELINE
# =======================
def main():
    print("📦 Loading raw data...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    all_forecasts = []

    # subset kecil dulu untuk efisiensi (bisa diperluas nanti)
    stores = df["store"].unique()[:3]
    items = df["item"].unique()[:5]

    for s in stores:
        for i in items:
            print(f"🔮 Generating forecast for Store {s}, Item {i}...")
            fc = generate_prophet_forecast(df, s, i)
            all_forecasts.append(fc)

    combined = pd.concat(all_forecasts, ignore_index=True)
    print("✨ Adding time-based features...")
    combined = add_time_features(combined)

    # simpan hasil hybrid dataset
    output_file = OUTDIR / "hybrid_train.csv"
    combined.to_csv(output_file, index=False)
    print(f"✅ Hybrid features saved to: {output_file}")


if __name__ == "__main__":
    main()
