"""
predict_future_demand.py
------------------------
Gunakan model Prophet + XGBoost (hybrid tuned)
untuk memprediksi permintaan (sales) 90 hari ke depan
pada tiap store dan item.

Output:
- CSV: reports/forecast/future_demand_forecast.csv
- Grafik: reports/forecast/plot_future_forecast_storeX_itemY.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
from prophet import Prophet
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# ===============================
# CONFIG
# ===============================
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "demand-forecasting-kernels-only" / "be-train.csv"
MODEL_PATH = ROOT / "models" / "hybrid_xgb_tuned.pkl"
OUTDIR = ROOT / "reports" / "forecast"
OUTDIR.mkdir(parents=True, exist_ok=True)

FORECAST_DAYS = 90  # periode prediksi ke depan

# ===============================
# FUNCTIONS
# ===============================
def add_time_features(df):
    """Tambahkan fitur waktu untuk data masa depan"""
    df["dayofweek"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    df["year"] = df["ds"].dt.year
    df["lag_1"] = df["yhat"].shift(1)
    df["lag_7"] = df["yhat"].shift(7)
    df["rolling_mean_7"] = df["yhat"].shift(1).rolling(7).mean()
    df["rolling_std_7"] = df["yhat"].shift(1).rolling(7).std()
    df = df.dropna()
    return df

def predict_future_for_store_item(df, store, item, xgb_model):
    """Prediksi masa depan untuk satu kombinasi store-item"""
    subset = df[(df["store"] == store) & (df["item"] == item)].copy().sort_values("date")
    train = subset.rename(columns={"date": "ds", "sales": "y"})

    # Latih Prophet pada seluruh data historis
    prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    prophet.fit(train)

    # Buat data masa depan
    future = prophet.make_future_dataframe(periods=FORECAST_DAYS)
    forecast = prophet.predict(future)
    merged = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    merged["store"] = store
    merged["item"] = item

    # Buat fitur untuk XGBoost
    hybrid_df = add_time_features(merged)
    X_future = hybrid_df[["yhat", "dayofweek", "month", "year", "lag_1", "lag_7", "rolling_mean_7", "rolling_std_7"]]

    # Prediksi residual dari XGBoost
    hybrid_df["xgb_correction"] = xgb_model.predict(X_future)
    hybrid_df["hybrid_forecast"] = hybrid_df["yhat"] + hybrid_df["xgb_correction"]

    return hybrid_df


def plot_future_forecast(df, store, item):
    """Buat grafik forecast masa depan"""
    plt.figure(figsize=(10, 5))
    plt.plot(df["ds"], df["yhat"], label="Prophet Forecast", color="royalblue")
    plt.plot(df["ds"], df["hybrid_forecast"], label="Hybrid (Prophet + XGBoost)", color="hotpink")
    plt.fill_between(df["ds"], df["yhat_lower"], df["yhat_upper"], color="lightblue", alpha=0.3)
    plt.title(f"Future Demand Forecast — Store {store}, Item {item}")
    plt.xlabel("Date")
    plt.ylabel("Predicted Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / f"plot_future_forecast_store{store}_item{item}.png", dpi=150)
    plt.close()


# ===============================
# MAIN
# ===============================
def main():
    print("📦 Loading historical data...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    xgb_model: XGBRegressor = joblib.load(MODEL_PATH)

    stores = df["store"].unique()[:3]   # subset untuk efisiensi (bisa diubah)
    items = df["item"].unique()[:5]

    all_results = []
    for s in stores:
        for i in items:
            print(f"🔮 Forecasting future demand for Store {s}, Item {i}...")
            fc = predict_future_for_store_item(df, s, i, xgb_model)
            all_results.append(fc)

            # Simpan grafik
            plot_future_forecast(fc, s, i)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUTDIR / "future_demand_forecast.csv", index=False)
    print(f"\n✅ Future forecasts saved to: {OUTDIR / 'future_demand_forecast.csv'}")
    print(f"📊 Plots saved to: {OUTDIR}")

if __name__ == "__main__":
    main()
