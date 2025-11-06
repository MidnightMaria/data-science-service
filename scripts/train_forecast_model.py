# scripts/train_forecast_model.py

from pathlib import Path
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from joblib import dump

# =============================
# CONFIGURATION
# =============================
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "train_features.csv"
OUTDIR = ROOT / "reports" / "forecast"
MODELDIR = ROOT / "models"
OUTDIR.mkdir(parents=True, exist_ok=True)
MODELDIR.mkdir(parents=True, exist_ok=True)

# Contoh data untuk satu kombinasi (bisa kamu ganti nanti)
EXAMPLE_STORE = 1
EXAMPLE_ITEM = 1

# =============================
# MAIN PIPELINE
# =============================
def main():
    print("📦 Loading processed dataset...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(df.head())

    # Ambil subset data satu store dan satu item
    one = df[(df["store"] == EXAMPLE_STORE) & (df["item"] == EXAMPLE_ITEM)].copy()

    # Prophet butuh format kolom: ds (datetime) dan y (nilai target)
    one = one.rename(columns={"date": "ds", "sales": "y"})[["ds", "y"]]
    print(f"\n📈 Data untuk Store {EXAMPLE_STORE}, Item {EXAMPLE_ITEM}")
    print(one.head())

    # =============================
    # 1️⃣ TRAINING MODEL
    # =============================
    print("\n🚀 Training Prophet model...")
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.5
    )
    model.fit(one)

    # =============================
    # 2️⃣ FORECASTING
    # =============================
    print("\n🔮 Forecasting future 90 days...")
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)

    # =============================
    # 3️⃣ VISUALIZATION
    # =============================
    print("\n📊 Plotting forecast results...")
    fig1 = model.plot(forecast)
    plt.title(f"Prophet Forecast — Store {EXAMPLE_STORE}, Item {EXAMPLE_ITEM}")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"plot_prophet_store{EXAMPLE_STORE}_item{EXAMPLE_ITEM}.png", dpi=150)

    fig2 = model.plot_components(forecast)
    plt.tight_layout()
    plt.savefig(OUTDIR / f"plot_components_store{EXAMPLE_STORE}_item{EXAMPLE_ITEM}.png", dpi=150)

    # =============================
    # 4️⃣ SAVE MODEL & FORECAST
    # =============================
    print("\n💾 Saving model and forecast results...")
    dump(model, MODELDIR / f"prophet_store{EXAMPLE_STORE}_item{EXAMPLE_ITEM}.joblib")
    forecast.to_csv(OUTDIR / f"forecast_store{EXAMPLE_STORE}_item{EXAMPLE_ITEM}.csv", index=False)

    print("\n✅ Training complete! Check your reports folder for plots and forecasts.")


if __name__ == "__main__":
    main()
