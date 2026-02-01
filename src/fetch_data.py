#!/usr/bin/env python3
"""
Fetch historical data for HSI forecasting model.
Sources: Yahoo Finance for HSI, SPX, NDX, USDCNH
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"

# Tickers to fetch
TICKERS = {
    "HSI": "^HSI",      # Hang Seng Index
    "SPX": "^GSPC",     # S&P 500
    "NDX": "^IXIC",     # Nasdaq Composite
    "USDCNH": "CNH=F",  # USD/CNH Futures (better data than CNH=X)
    "VIX": "^VIX",      # CBOE Volatility Index (fear gauge)
    "FXI": "FXI",       # iShares China Large Cap ETF (US traded - overnight HK proxy!)
}


def fetch_ticker(ticker: str, symbol: str, years: int = 3) -> pd.DataFrame:
    """Fetch OHLCV data for a single ticker."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365 + 30)  # Extra buffer
    
    print(f"Fetching {ticker} ({symbol})...")
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    
    # Flatten multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Keep only needed columns
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = [f"{ticker}_{col}".lower() for col in df.columns]
    
    print(f"  → {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
    return df


def fetch_all(years: int = 3) -> pd.DataFrame:
    """Fetch all tickers and merge on date."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    dfs = []
    for ticker, symbol in TICKERS.items():
        df = fetch_ticker(ticker, symbol, years)
        # Save individual raw files
        df.to_csv(RAW_DATA_DIR / f"{ticker.lower()}.csv")
        dfs.append(df)
    
    # Merge all on date index
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df, how="outer")
    
    # Forward fill gaps (e.g., HK holidays vs US holidays)
    merged = merged.ffill()
    
    # Keep only days where HSI traded
    merged = merged.dropna(subset=["hsi_close"])
    
    # Save merged data
    merged.to_csv(RAW_DATA_DIR / "merged.csv")
    print(f"\nMerged data: {len(merged)} rows saved to {RAW_DATA_DIR / 'merged.csv'}")
    
    return merged


if __name__ == "__main__":
    df = fetch_all(years=3)
    print("\nSample data:")
    print(df.tail())
