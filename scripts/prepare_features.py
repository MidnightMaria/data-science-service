# ============================================
# scripts/prepare_features.py
# ============================================
from pathlib import Path
import pandas as pd
import numpy as np

# ============ CONFIG =============
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "demand-forecasting-kernels-only" / "be-train.csv"
OUTDIR = ROOT / "data" / "processed"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============ 1️⃣ LOAD DATA ============
print("📦 Loading dataset...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values(["store", "item", "date"])
print(df.head())

# ============ 2️⃣ FEATURE ENGINEERING ============
print("✨ Creating time-based features...")

# Time-based features
df["dayofweek"] = df["date"].dt.dayofweek   # 0=Mon, 6=Sun
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df["dayofmonth"] = df["date"].dt.day
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

# Sin/Cos encoding for periodicity
df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
df["sin_dow"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["cos_dow"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

# ============ 3️⃣ CREATE LAGS ============
print("🔁 Creating lag and rolling features...")
df["lag_1"] = df.groupby(["store", "item"])["sales"].shift(1)
df["lag_7"] = df.groupby(["store", "item"])["sales"].shift(7)
df["lag_14"] = df.groupby(["store", "item"])["sales"].shift(14)

# Rolling mean and std (7-day window)
print("📊 Computing rolling statistics...")
df["rolling_mean_7"] = (
    df.groupby(["store", "item"])["sales"]
      .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
)

df["rolling_std_7"] = (
    df.groupby(["store", "item"])["sales"]
      .transform(lambda x: x.shift(1).rolling(7, min_periods=1).std())
)

# Drop NA dari lag
df = df.dropna().reset_index(drop=True)

# ============ 4️⃣ TRAIN-TEST SPLIT ============
print("🧩 Splitting train/test data...")
train = df[df["date"] < "2017-10-01"]
test  = df[df["date"] >= "2017-10-01"]

print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# ============ 5️⃣ SAVE ============
train.to_csv(OUTDIR / "train_features.csv", index=False)
test.to_csv(OUTDIR / "test_features.csv", index=False)

print("✅ Features prepared successfully!")
print("Saved to:", OUTDIR)
