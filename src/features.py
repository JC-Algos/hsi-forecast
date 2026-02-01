#!/usr/bin/env python3
"""
Feature engineering for HSI forecasting model.
Creates features from raw OHLCV data and targets for training.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def calculate_ma(series: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return series.rolling(window=window).mean()


def calculate_volatility(series: pd.Series, window: int = 10) -> pd.Series:
    """Calculate rolling standard deviation of returns."""
    returns = series.pct_change()
    return returns.rolling(window=window).std()


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all features for the model.
    
    Features (all using YESTERDAY's data to predict TODAY):
    - HSI_close: Previous day close
    - HSI_change_pct: Previous day % change
    - HSI_EMA20: 20-day EMA of close
    - HSI_MA5: 5-day MA of close
    - HSI_volatility: 10-day rolling volatility
    - SPX_close: S&P 500 previous close
    - SPX_change_pct: S&P 500 overnight change (captures gap risk)
    - NDX_close: Nasdaq previous close
    - NDX_change_pct: Nasdaq overnight change (captures gap risk)
    - USDCNH: USD/CNH previous close
    - USDCNH_change_pct: USD/CNH change (currency risk)
    - VIX: CBOE Volatility Index (fear gauge)
    - day_of_week: 0-4 (Mon-Fri)
    """
    features = pd.DataFrame(index=df.index)
    
    # HSI features
    features["hsi_prev_close"] = df["hsi_close"].shift(1)
    features["hsi_change_pct"] = df["hsi_close"].pct_change().shift(1)
    features["hsi_ema20"] = calculate_ema(df["hsi_close"], 20).shift(1)
    features["hsi_ma5"] = calculate_ma(df["hsi_close"], 5).shift(1)
    features["hsi_volatility"] = calculate_volatility(df["hsi_close"], 10).shift(1)
    
    # External features (use previous day's close)
    features["spx_close"] = df["spx_close"].shift(1)
    features["spx_change_pct"] = df["spx_close"].pct_change().shift(1)  # Overnight US move
    features["ndx_close"] = df["ndx_close"].shift(1)
    features["ndx_change_pct"] = df["ndx_close"].pct_change().shift(1)  # Overnight US tech move
    features["usdcnh"] = df["usdcnh_close"].shift(1)
    features["usdcnh_change_pct"] = df["usdcnh_close"].pct_change().shift(1)  # Currency move
    
    # VIX - fear gauge (high VIX = expect bigger moves/gaps)
    if "vix_close" in df.columns:
        features["vix"] = df["vix_close"].shift(1)
        features["vix_change_pct"] = df["vix_close"].pct_change().shift(1)
    
    # FXI - China Large Cap ETF (US traded, best overnight HK proxy!)
    # FXI closes at 4AM HKT, directly reflects overnight sentiment
    if "fxi_close" in df.columns:
        features["fxi_close"] = df["fxi_close"].shift(1)
        features["fxi_change_pct"] = df["fxi_close"].pct_change().shift(1)
    
    # Day of week (0=Monday, 4=Friday)
    features["day_of_week"] = df.index.dayofweek
    
    return features


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target variables for TODAY's prediction.
    
    Targets:
    - high_pct: Today's high as % from today's open
    - low_pct: Today's low as % from today's open
    - direction: 1 if close > open, else 0
    """
    targets = pd.DataFrame(index=df.index)
    
    # Range as % from open
    targets["high_pct"] = (df["hsi_high"] - df["hsi_open"]) / df["hsi_open"] * 100
    targets["low_pct"] = (df["hsi_low"] - df["hsi_open"]) / df["hsi_open"] * 100
    
    # Direction: 1 if up (close > open), 0 if down
    targets["direction"] = (df["hsi_close"] > df["hsi_open"]).astype(int)
    
    # Also store open for later use
    targets["hsi_open"] = df["hsi_open"]
    
    return targets


def prepare_dataset(years: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw data, create features and targets, save processed data.
    
    Returns:
        features: DataFrame with input features
        targets: DataFrame with target variables
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load merged raw data
    raw_path = RAW_DATA_DIR / "merged.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Run fetch_data.py first: {raw_path}")
    
    df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} rows from {raw_path}")
    
    # Create features and targets
    features = create_features(df)
    targets = create_targets(df)
    
    # Combine and drop rows with NaN (from rolling calculations)
    combined = features.join(targets)
    combined = combined.dropna()
    
    # Split back
    feature_cols = features.columns.tolist()
    target_cols = targets.columns.tolist()
    
    features = combined[feature_cols]
    targets = combined[target_cols]
    
    # Save processed data
    features.to_csv(PROCESSED_DIR / "features.csv")
    targets.to_csv(PROCESSED_DIR / "targets.csv")
    
    print(f"\nProcessed data: {len(features)} samples")
    print(f"Features saved to: {PROCESSED_DIR / 'features.csv'}")
    print(f"Targets saved to: {PROCESSED_DIR / 'targets.csv'}")
    
    # Print feature summary
    print("\nFeature columns:")
    for col in features.columns:
        print(f"  - {col}")
    
    print("\nTarget columns:")
    for col in targets.columns:
        print(f"  - {col}")
    
    return features, targets


if __name__ == "__main__":
    features, targets = prepare_dataset()
    
    print("\nFeature sample:")
    print(features.tail())
    
    print("\nTarget sample:")
    print(targets.tail())
    
    print("\nTarget statistics:")
    print(targets[["high_pct", "low_pct", "direction"]].describe())
