#!/usr/bin/env python3
"""
Feature engineering v2 - includes Stock Connect (Southbound) data
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


def load_price_data() -> pd.DataFrame:
    """Load merged price data."""
    return pd.read_csv(RAW_DIR / "merged.csv", index_col=0, parse_dates=True)


def load_stock_connect() -> pd.DataFrame:
    """Load Stock Connect (Southbound) data."""
    path = RAW_DIR / "stock_connect.csv"
    if not path.exists():
        print("⚠️ Stock Connect data not found. Run fetch_stock_connect.py first.")
        return pd.DataFrame()
    
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def calculate_features_v2(df: pd.DataFrame, stock_connect: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate features including Stock Connect flow.
    
    New features:
    - southbound_net: Daily net flow (millions HKD)
    - southbound_net_pct: Net flow as % of total turnover
    - southbound_net_5d_avg: 5-day moving average of net flow
    - southbound_flow_momentum: Change in net flow vs yesterday
    """
    features = pd.DataFrame(index=df.index)
    
    # === Original price features ===
    features["hsi_prev_close"] = df["hsi_close"]
    features["hsi_change_pct"] = df["hsi_close"].pct_change()
    features["hsi_ema20"] = df["hsi_close"].ewm(span=20, adjust=False).mean()
    features["hsi_ma5"] = df["hsi_close"].rolling(5).mean()
    features["hsi_volatility"] = df["hsi_close"].pct_change().rolling(10).std()
    
    features["spx_close"] = df["spx_close"]
    features["spx_change_pct"] = df["spx_close"].pct_change()
    
    features["ndx_close"] = df["ndx_close"]
    features["ndx_change_pct"] = df["ndx_close"].pct_change()
    
    features["usdcnh"] = df["usdcnh_close"]
    features["usdcnh_change_pct"] = df["usdcnh_close"].pct_change()
    
    features["vix"] = df["vix_close"]
    features["vix_change_pct"] = df["vix_close"].pct_change()
    
    features["fxi_close"] = df["fxi_close"]
    features["fxi_change_pct"] = df["fxi_close"].pct_change()
    
    features["day_of_week"] = df.index.dayofweek
    
    # === Stock Connect features ===
    if len(stock_connect) > 0:
        # Merge stock connect data
        sc = stock_connect[['southbound_net', 'total_southbound_buy', 'total_southbound_sell']].copy()
        
        # Align indices
        features = features.join(sc, how='left')
        
        # Fill missing (weekends, holidays) with forward fill then 0
        features['southbound_net'] = features['southbound_net'].fillna(method='ffill').fillna(0)
        
        # Net flow as percentage of total turnover
        total_turnover = features['total_southbound_buy'].fillna(0) + features['total_southbound_sell'].fillna(0)
        features['southbound_net_pct'] = features['southbound_net'] / total_turnover.replace(0, np.nan)
        features['southbound_net_pct'] = features['southbound_net_pct'].fillna(0)
        
        # 5-day moving average of net flow
        features['southbound_net_5d_avg'] = features['southbound_net'].rolling(5).mean()
        
        # Flow momentum (change vs yesterday)
        features['southbound_flow_momentum'] = features['southbound_net'].diff()
        
        # Consecutive days of inflow/outflow
        features['southbound_direction'] = np.sign(features['southbound_net'])
        
        # Drop intermediate columns
        features = features.drop(columns=['total_southbound_buy', 'total_southbound_sell'], errors='ignore')
    else:
        print("⚠️ No Stock Connect data - using zeros")
        features['southbound_net'] = 0
        features['southbound_net_pct'] = 0
        features['southbound_net_5d_avg'] = 0
        features['southbound_flow_momentum'] = 0
        features['southbound_direction'] = 0
    
    return features


def calculate_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate prediction targets."""
    targets = pd.DataFrame(index=df.index)
    
    prev_close = df["hsi_close"].shift(1)
    targets["high_pct"] = (df["hsi_high"] - prev_close) / prev_close * 100
    targets["low_pct"] = (df["hsi_low"] - prev_close) / prev_close * 100
    targets["direction"] = (df["hsi_close"] > prev_close).astype(int)
    
    return targets


def main():
    """Generate features with Stock Connect data."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading price data...")
    df = load_price_data()
    print(f"  Loaded {len(df)} rows: {df.index[0].date()} to {df.index[-1].date()}")
    
    print("\nLoading Stock Connect data...")
    stock_connect = load_stock_connect()
    if len(stock_connect) > 0:
        print(f"  Loaded {len(stock_connect)} rows: {stock_connect.index[0].date()} to {stock_connect.index[-1].date()}")
    
    print("\nCalculating features...")
    features = calculate_features_v2(df, stock_connect)
    
    print("Calculating targets...")
    targets = calculate_targets(df)
    
    # Align and drop NaN
    common_idx = features.dropna().index.intersection(targets.dropna().index)
    features = features.loc[common_idx]
    targets = targets.loc[common_idx]
    
    print(f"\nFinal dataset: {len(features)} samples")
    print(f"Features: {list(features.columns)}")
    
    # Save
    features.to_csv(PROCESSED_DIR / "features_v2.csv")
    targets.to_csv(PROCESSED_DIR / "targets_v2.csv")
    
    print(f"\n✅ Saved to {PROCESSED_DIR}")
    
    # Show Stock Connect feature stats
    if 'southbound_net' in features.columns:
        print("\n📊 Stock Connect Feature Stats:")
        print(f"  southbound_net mean: {features['southbound_net'].mean():,.0f}M HKD")
        print(f"  southbound_net std:  {features['southbound_net'].std():,.0f}M HKD")
        print(f"  Days with inflow:    {(features['southbound_net'] > 0).sum()}")
        print(f"  Days with outflow:   {(features['southbound_net'] < 0).sum()}")


if __name__ == "__main__":
    main()
