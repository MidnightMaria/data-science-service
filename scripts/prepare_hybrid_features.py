"""
prepare_hybrid_features.py
--------------------------
Menyiapkan dataset hybrid forecasting tanpa data leakage.

Output:
- data/processed/hybrid_train.csv
- data/processed/hybrid_val.csv
- data/processed/hybrid_test.csv

Konsep:
1. Untuk setiap (store, item), split data menjadi:
   - train
   - validation
   - test
2. Fit Prophet hanya pada data train
3. Buat prediksi Prophet untuk train, val, test
4. Hitung residual = sales - yhat
5. Tambahkan fitur time series yang kausal (hanya pakai masa lalu)
6. Gabungkan semua series menjadi dataset global untuk XGBoost
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from prophet import Prophet

warnings.filterwarnings("ignore")

# =======================
# CONFIG
# =======================
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "demand-forecasting-kernels-only" / "be-train.csv"
OUTDIR = ROOT / "data" / "processed"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Split horizon
VAL_DAYS = 90
TEST_DAYS = 90

# Minimal data length supaya split aman
MIN_REQUIRED_LENGTH = VAL_DAYS + TEST_DAYS + 30


def safe_prophet_forecast(train_df: pd.DataFrame, all_dates: pd.Series) -> pd.DataFrame:
    """
    Fit Prophet hanya di train, lalu prediksi untuk seluruh tanggal yang dibutuhkan.
    train_df harus punya kolom: date, sales
    all_dates adalah seluruh tanggal pada series tsb.
    """
    prophet_train = train_df.rename(columns={"date": "ds", "sales": "y"})[["ds", "y"]].copy()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.5,
    )
    model.fit(prophet_train)

    future = pd.DataFrame({"ds": pd.to_datetime(all_dates)})
    forecast = model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    return forecast


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan fitur yang konsisten untuk XGBoost.
    Semua fitur harus kausal: hanya memakai informasi masa lalu.
    """
    out = df.copy().sort_values(["store", "item", "date"]).reset_index(drop=True)

    # Calendar features
    out["dayofweek"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["year"] = out["date"].dt.year
    out["dayofmonth"] = out["date"].dt.day
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)

    # ID features untuk global model
    out["store_id"] = out["store"].astype(int)
    out["item_id"] = out["item"].astype(int)

    # Prophet-based features
    out["lag_yhat_1"] = out.groupby(["store", "item"])["yhat"].shift(1)
    out["lag_yhat_7"] = out.groupby(["store", "item"])["yhat"].shift(7)
    out["rolling_yhat_mean_7"] = (
        out.groupby(["store", "item"])["yhat"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=7).mean())
    )
    out["rolling_yhat_std_7"] = (
        out.groupby(["store", "item"])["yhat"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=7).std())
    )

    # Sales-based lag features (masih kausal, karena shift ke belakang)
    out["lag_sales_1"] = out.groupby(["store", "item"])["sales"].shift(1)
    out["lag_sales_7"] = out.groupby(["store", "item"])["sales"].shift(7)
    out["rolling_sales_mean_7"] = (
        out.groupby(["store", "item"])["sales"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=7).mean())
    )
    out["rolling_sales_std_7"] = (
        out.groupby(["store", "item"])["sales"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=7).std())
    )

    return out


def split_series(df_one: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split satu series menjadi train / val / test.
    """
    df_one = df_one.sort_values("date").reset_index(drop=True)

    train = df_one.iloc[: -(VAL_DAYS + TEST_DAYS)].copy()
    val = df_one.iloc[-(VAL_DAYS + TEST_DAYS) : -TEST_DAYS].copy()
    test = df_one.iloc[-TEST_DAYS:].copy()

    return train, val, test


def process_one_series(df: pd.DataFrame, store: int, item: int) -> pd.DataFrame | None:
    """
    Proses satu kombinasi (store, item):
    - split
    - fit Prophet hanya di train
    - prediksi untuk seluruh tanggal
    - hitung residual
    - tandai split
    """
    subset = (
        df[(df["store"] == store) & (df["item"] == item)]
        .copy()
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(subset) < MIN_REQUIRED_LENGTH:
        print(f"Skipping store={store}, item={item}: insufficient length ({len(subset)})")
        return None

    train, val, test = split_series(subset)

    # Prophet fit hanya di train
    forecast = safe_prophet_forecast(train_df=train, all_dates=subset["date"])

    merged = subset.merge(forecast, left_on="date", right_on="ds", how="left")
    merged.drop(columns=["ds"], inplace=True)

    # Residual dihitung untuk semua baris, tetapi:
    # - training XGBoost nanti hanya pakai split='train'
    # - val/test hanya untuk evaluasi
    merged["residual"] = merged["sales"] - merged["yhat"]

    # Tandai split
    merged["split"] = "train"
    merged.loc[merged["date"].isin(val["date"]), "split"] = "val"
    merged.loc[merged["date"].isin(test["date"]), "split"] = "test"

    return merged


def main():
    print("Loading raw dataset...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)

    expected_cols = {"date", "store", "item", "sales"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    stores = sorted(df["store"].unique())
    items = sorted(df["item"].unique())

    all_series = []

    print(f"Preparing hybrid features for {len(stores)} stores x {len(items)} items...")
    for s in stores:
        for i in items:
            print(f"Processing store={s}, item={i} ...")
            result = process_one_series(df, s, i)
            if result is not None:
                all_series.append(result)

    if not all_series:
        raise RuntimeError("No valid series processed.")

    combined = pd.concat(all_series, ignore_index=True)
    combined = combined.sort_values(["store", "item", "date"]).reset_index(drop=True)

    # Tambahkan fitur kausal
    combined = add_time_features(combined)

    # Drop baris awal yang belum punya lag/rolling lengkap
    feature_cols_required = [
        "yhat",
        "dayofweek",
        "month",
        "year",
        "dayofmonth",
        "is_weekend",
        "store_id",
        "item_id",
        "lag_yhat_1",
        "lag_yhat_7",
        "rolling_yhat_mean_7",
        "rolling_yhat_std_7",
        "lag_sales_1",
        "lag_sales_7",
        "rolling_sales_mean_7",
        "rolling_sales_std_7",
        "residual",
    ]
    combined = combined.dropna(subset=feature_cols_required).reset_index(drop=True)

    # Split final datasets
    hybrid_train = combined[combined["split"] == "train"].copy()
    hybrid_val = combined[combined["split"] == "val"].copy()
    hybrid_test = combined[combined["split"] == "test"].copy()

    # Simpan
    train_path = OUTDIR / "hybrid_train.csv"
    val_path = OUTDIR / "hybrid_val.csv"
    test_path = OUTDIR / "hybrid_test.csv"

    hybrid_train.to_csv(train_path, index=False)
    hybrid_val.to_csv(val_path, index=False)
    hybrid_test.to_csv(test_path, index=False)

    print("\nDone.")
    print(f"hybrid_train saved to: {train_path} | rows={len(hybrid_train):,}")
    print(f"hybrid_val saved to:   {val_path} | rows={len(hybrid_val):,}")
    print(f"hybrid_test saved to:  {test_path} | rows={len(hybrid_test):,}")

    # Ringkasan cek
    print("\nSplit summary:")
    print(combined["split"].value_counts())

    print("\nColumns:")
    print(combined.columns.tolist())


if __name__ == "__main__":
    main()