# scripts/evaluate_model.py

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# =============================
# CONFIGURATION
# =============================
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT / "data" / "processed" / "train_features.csv"
OUTDIR = ROOT / "reports" / "evaluation"
MODELDIR = ROOT / "models"
OUTDIR.mkdir(parents=True, exist_ok=True)

EXAMPLE_STORE = 1
EXAMPLE_ITEM = 1
MODEL_PATH = MODELDIR / f"prophet_store{EXAMPLE_STORE}_item{EXAMPLE_ITEM}.joblib"

# =============================
# MAIN FUNCTION
# =============================
def main():
    print("📦 Loading data & model...")

    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    model = load(MODEL_PATH)

    # Subset sesuai store dan item
    one = df[(df["store"] == EXAMPLE_STORE) & (df["item"] == EXAMPLE_ITEM)].copy()
    one = one.rename(columns={"date": "ds", "sales": "y"})[["ds", "y"]]

    print(f"📊 Evaluating Store {EXAMPLE_STORE}, Item {EXAMPLE_ITEM}...")

    # Split 90 hari terakhir untuk testing
    train = one.iloc[:-90]
    test = one.iloc[-90:]

    # Re-train Prophet di data training saja (fresh instance)
    print("🚀 Training fresh Prophet model for evaluation...")
    from prophet import Prophet
    eval_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.5
    )
    eval_model.fit(train)


    # Forecast periode test
    future = eval_model.make_future_dataframe(periods=90)
    forecast = eval_model.predict(future)


    # Gabungkan hasil aktual dan prediksi
    pred = forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]].join(test.set_index("ds"))
    pred = pred.dropna()

    # =============================
    # EVALUATION METRICS
    # =============================
    mae = mean_absolute_error(pred["y"], pred["yhat"])
    rmse = np.sqrt(mean_squared_error(pred["y"], pred["yhat"]))
    mape = np.mean(np.abs((pred["y"] - pred["yhat"]) / pred["y"])) * 100

    metrics = pd.DataFrame({
        "Store": [EXAMPLE_STORE],
        "Item": [EXAMPLE_ITEM],
        "MAE": [mae],
        "RMSE": [rmse],
        "MAPE (%)": [mape]
    })

    metrics.to_csv(OUTDIR / f"metrics_store{EXAMPLE_STORE}_item{EXAMPLE_ITEM}.csv", index=False)

    print("\n✅ Evaluation complete!")
    print(metrics)

    # =============================
    # VISUALIZATION
    # =============================
    plt.figure(figsize=(10, 5))
    plt.plot(pred.index, pred["y"], label="Actual", color="black")
    plt.plot(pred.index, pred["yhat"], label="Forecast", color="royalblue")
    plt.fill_between(pred.index, pred["yhat_lower"], pred["yhat_upper"], color="skyblue", alpha=0.3)
    plt.legend()
    plt.title(f"Forecast vs Actual — Store {EXAMPLE_STORE}, Item {EXAMPLE_ITEM}")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"plot_forecast_vs_actual_store{EXAMPLE_STORE}_item{EXAMPLE_ITEM}.png", dpi=150)
    plt.close()

    print(f"📁 Metrics & plots saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
