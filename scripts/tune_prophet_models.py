"""
tune_prophet_models.py
----------------------
Re-train Prophet untuk item dengan performa buruk (MAPE > 25%)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "demand-forecasting-kernels-only" / "be-train.csv"
METRICS_PATH = ROOT / "reports" / "evaluation" / "summary_metrics.csv"
OUTDIR = ROOT / "reports" / "tuned_evaluation"
OUTDIR.mkdir(parents=True, exist_ok=True)

VAL_DAYS = 90

def evaluate_prophet(df, store, item, **params):
    subset = df[(df["store"] == store) & (df["item"] == item)].copy().sort_values("date")
    train = subset.iloc[:-VAL_DAYS]
    val = subset.iloc[-VAL_DAYS:]
    train = train.rename(columns={"date":"ds","sales":"y"})
    
    model = Prophet(**params)
    model.fit(train)
    
    future = model.make_future_dataframe(periods=VAL_DAYS)
    forecast = model.predict(future)
    y_pred = forecast.set_index("ds").loc[val["date"], "yhat"].values
    y_true = val["sales"].values
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred)/y_true)) * 100
    
    return mae, rmse, mape


def main():
    df = pd.read_csv(RAW_PATH, parse_dates=["date"])
    metrics = pd.read_csv(METRICS_PATH)
    to_tune = metrics[metrics["MAPE (%)"] > 25]

    print(f"🔧 Menemukan {len(to_tune)} kombinasi yang perlu di-tune...")

    tuned_results = []
    for _, row in to_tune.iterrows():
        s, i = row["store"], row["item"]
        print(f"⚙️ Tuning Store {s}, Item {i}...")

        mae, rmse, mape = evaluate_prophet(
            df, s, i,
            yearly_seasonality=True,
            weekly_seasonality=True,
            changepoint_prior_scale=1.0,
            seasonality_mode="multiplicative"
        )

        tuned_results.append({
            "store": s, "item": i,
            "MAE_tuned": mae, "RMSE_tuned": rmse, "MAPE_tuned (%)": mape
        })
        print(f"✅ MAPE baru: {mape:.2f}%")

    tuned_df = pd.DataFrame(tuned_results)
    tuned_df.to_csv(OUTDIR / "tuned_metrics.csv", index=False)
    print(f"\n✨ Selesai! Disimpan ke {OUTDIR / 'tuned_metrics.csv'}")


if __name__ == "__main__":
    main()
