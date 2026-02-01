#!/usr/bin/env python3
"""
Daily prediction for HSI range and direction.
Run before market open to get today's forecast.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from datetime import datetime, timedelta
import json
import argparse

PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = PROJECT_DIR / "models"


def fetch_recent_data(days: int = 30) -> pd.DataFrame:
    """Fetch recent data needed for feature calculation."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    tickers = {
        "HSI": "^HSI",
        "SPX": "^GSPC",
        "NDX": "^IXIC",
        "USDCNH": "CNH=F",  # USD/CNH Futures
        "VIX": "^VIX",      # Fear gauge
        "FXI": "FXI",       # China Large Cap ETF (overnight HK proxy)
    }
    
    dfs = []
    for name, symbol in tickers.items():
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close"]].copy()
        df.columns = [f"{name.lower()}_{col.lower()}" for col in df.columns]
        dfs.append(df)
    
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df, how="outer")
    
    merged = merged.ffill()
    merged = merged.dropna()
    
    return merged


def calculate_features(df: pd.DataFrame) -> pd.Series:
    """Calculate features for the most recent day."""
    # EMA and MA
    df["hsi_ema20"] = df["hsi_close"].ewm(span=20, adjust=False).mean()
    df["hsi_ma5"] = df["hsi_close"].rolling(5).mean()
    df["hsi_volatility"] = df["hsi_close"].pct_change().rolling(10).std()
    
    # Get the latest row (yesterday's data for today's prediction)
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    features = pd.Series({
        "hsi_prev_close": latest["hsi_close"],
        "hsi_change_pct": (latest["hsi_close"] - prev["hsi_close"]) / prev["hsi_close"],
        "hsi_ema20": latest["hsi_ema20"],
        "hsi_ma5": latest["hsi_ma5"],
        "hsi_volatility": latest["hsi_volatility"],
        "spx_close": latest["spx_close"],
        "spx_change_pct": (latest["spx_close"] - prev["spx_close"]) / prev["spx_close"],
        "ndx_close": latest["ndx_close"],
        "ndx_change_pct": (latest["ndx_close"] - prev["ndx_close"]) / prev["ndx_close"],
        "usdcnh": latest["usdcnh_close"],
        "usdcnh_change_pct": (latest["usdcnh_close"] - prev["usdcnh_close"]) / prev["usdcnh_close"],
        "vix": latest["vix_close"] if "vix_close" in df.columns else 20.0,
        "vix_change_pct": (latest["vix_close"] - prev["vix_close"]) / prev["vix_close"] if "vix_close" in df.columns else 0.0,
        "fxi_close": latest["fxi_close"] if "fxi_close" in df.columns else 40.0,
        "fxi_change_pct": (latest["fxi_close"] - prev["fxi_close"]) / prev["fxi_close"] if "fxi_close" in df.columns else 0.0,
        "day_of_week": datetime.now().weekday(),  # Today's day of week
    })
    
    return features


def load_models():
    """Load trained XGBoost models."""
    model_high = xgb.XGBRegressor()
    model_high.load_model(MODELS_DIR / "range_high.json")
    
    model_low = xgb.XGBRegressor()
    model_low.load_model(MODELS_DIR / "range_low.json")
    
    model_dir = xgb.XGBClassifier()
    model_dir.load_model(MODELS_DIR / "direction.json")
    
    return model_high, model_low, model_dir


def calculate_volatility_multiplier(df: pd.DataFrame, lookback: int = 20) -> tuple[float, str, float]:
    """
    Calculate volatility multiplier based on current vs historical volatility.
    
    Returns:
        multiplier: float (1.0 = normal, >1 = high vol, <1 = low vol)
        regime: str ('LOW', 'NORMAL', 'HIGH', 'EXTREME')
        vol_ratio: float
    """
    # Calculate recent volatility (10-day)
    returns = df["hsi_close"].pct_change()
    recent_vol = returns.iloc[-10:].std()
    
    # Calculate long-term average volatility
    long_term_vol = returns.std()
    
    # Volatility ratio
    vol_ratio = recent_vol / long_term_vol if long_term_vol > 0 else 1.0
    
    # Determine regime and multiplier
    if vol_ratio < 0.7:
        regime = "LOW"
        multiplier = 0.85
    elif vol_ratio < 1.0:
        regime = "NORMAL"
        multiplier = 1.0
    elif vol_ratio < 1.5:
        regime = "HIGH"
        multiplier = 1.2
    else:
        regime = "EXTREME"
        multiplier = 1.5
    
    return multiplier, regime, vol_ratio


def calculate_gap_adjustment(features: pd.Series) -> tuple[float, float, str]:
    """
    Calculate gap adjustment based on FXI (overnight HK proxy), VIX and US moves.
    
    FXI (China Large Cap ETF) is the best overnight indicator - trades in US hours.
    When FXI is down → expect HK to gap down
    When FXI is up → expect HK to gap up
    
    Returns:
        high_adj: adjustment to high prediction (%)
        low_adj: adjustment to low prediction (%)
        reason: explanation string
    """
    vix = features.get("vix", 20.0)
    vix_change = features.get("vix_change_pct", 0.0)
    spx_change = features.get("spx_change_pct", 0.0)
    ndx_change = features.get("ndx_change_pct", 0.0)
    fxi_change = features.get("fxi_change_pct", 0.0)
    
    us_avg_change = (spx_change + ndx_change) / 2
    
    high_adj = 0.0
    low_adj = 0.0
    reasons = []
    
    # FXI is the BEST overnight indicator for HK
    # FXI change directly translates to expected HK gap
    if abs(fxi_change) > 0.005:  # FXI moved > 0.5%
        # FXI change is ~1:1 with expected HK gap
        if fxi_change < 0:
            high_adj = fxi_change * 0.8  # FXI down → reduce high expectation
            low_adj = fxi_change * 1.5   # FXI down → widen low (gap down)
            reasons.append(f"FXI {fxi_change*100:+.1f}% → gap down")
        else:
            high_adj = fxi_change * 1.2  # FXI up → raise high
            low_adj = fxi_change * 0.5   # FXI up → raise low (gap up)
            reasons.append(f"FXI {fxi_change*100:+.1f}% → gap up")
    
    # Additional VIX adjustment
    if vix_change > 0.02 and us_avg_change < -0.003:
        low_adj += us_avg_change * 1.5  # Additional downside from fear
        reasons.append(f"VIX↑ + US↓")
    
    # High VIX level → widen range slightly
    if vix > 20:
        range_mult = 1 + (vix - 20) / 100
        if high_adj != 0 or low_adj != 0:
            high_adj *= range_mult
            low_adj *= range_mult
        reasons.append(f"VIX {vix:.0f}")
    
    reason = " | ".join(reasons) if reasons else "Normal"
    
    return high_adj, low_adj, reason


def predict_today(output_format: str = "json") -> dict:
    """
    Make prediction for today's HSI range and direction.
    
    Returns:
        dict with prediction results
    """
    # Fetch data and calculate features
    df = fetch_recent_data(days=30)
    features = calculate_features(df)
    
    # Load models
    model_high, model_low, model_dir = load_models()
    
    # Get feature order from metadata
    with open(MODELS_DIR / "metadata.json") as f:
        metadata = json.load(f)
    feature_cols = metadata["feature_cols"]
    
    # Prepare input
    X = features[feature_cols].values.reshape(1, -1)
    
    # Predict base values
    high_pct_base = model_high.predict(X)[0]
    low_pct_base = model_low.predict(X)[0]
    direction_prob = model_dir.predict_proba(X)[0, 1]
    
    # Calculate volatility multiplier
    vol_multiplier, vol_regime, vol_ratio = calculate_volatility_multiplier(df)
    
    # Calculate gap adjustment based on VIX and overnight US moves
    gap_high_adj, gap_low_adj, gap_reason = calculate_gap_adjustment(features)
    
    # Apply volatility adjustment to range (scale from center)
    range_center = (high_pct_base + low_pct_base) / 2
    high_pct = range_center + (high_pct_base - range_center) * vol_multiplier
    low_pct = range_center + (low_pct_base - range_center) * vol_multiplier
    
    # Apply gap adjustment
    high_pct += gap_high_adj
    low_pct += gap_low_adj
    
    # Use yesterday's close as reference (we don't know today's open yet!)
    yesterday_close = features["hsi_prev_close"]
    
    # Calculate absolute levels from yesterday's close
    predicted_high = yesterday_close * (1 + high_pct / 100)
    predicted_low = yesterday_close * (1 + low_pct / 100)
    
    result = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "prediction_time": datetime.now().isoformat(),
        "reference_price": float(yesterday_close),  # Yesterday's close as reference
        "predicted_high_pct": float(round(high_pct, 3)),
        "predicted_low_pct": float(round(low_pct, 3)),
        "predicted_high": int(round(predicted_high)),
        "predicted_low": int(round(predicted_low)),
        "predicted_range": int(round(predicted_high - predicted_low)),
        "predicted_range_pct": float(round(high_pct - low_pct, 3)),
        "direction_prob": float(round(direction_prob, 3)),
        "direction": "UP" if direction_prob > 0.5 else "DOWN",
        "confidence": float(round(abs(direction_prob - 0.5) * 2, 3)),  # 0-1 scale
        "volatility_regime": vol_regime,
        "volatility_multiplier": float(round(vol_multiplier, 2)),
        "volatility_ratio": float(round(vol_ratio, 2)),
        "gap_adjustment": gap_reason,
        "gap_high_adj": float(round(gap_high_adj, 3)),
        "gap_low_adj": float(round(gap_low_adj, 3)),
    }
    
    return result


def format_telegram(result: dict) -> str:
    """Format prediction for Telegram message."""
    direction_emoji = "📈" if result["direction"] == "UP" else "📉"
    confidence_bar = "█" * int(result["confidence"] * 10) + "░" * (10 - int(result["confidence"] * 10))
    
    # Volatility regime emoji
    vol_emoji = {
        "LOW": "😴",
        "NORMAL": "📊",
        "HIGH": "⚡",
        "EXTREME": "🔥"
    }.get(result["volatility_regime"], "📊")
    
    msg = f"""🎯 **HSI Daily Forecast** ({result['date']})

{direction_emoji} **Direction: {result['direction']}** ({result['direction_prob']:.1%})
Confidence: [{confidence_bar}] {result['confidence']:.1%}

📊 **Predicted Range (from prev close):**
• High: {result['predicted_high']:,} (+{result['predicted_high_pct']:.2f}%)
• Low: {result['predicted_low']:,} ({result['predicted_low_pct']:.2f}%)
• Range: {result['predicted_range']:,} pts ({result['predicted_range_pct']:.2f}%)

{vol_emoji} **Volatility: {result['volatility_regime']}** (×{result['volatility_multiplier']:.1f})

📍 Reference (Prev Close): {result['reference_price']:,.0f}
"""
    return msg


def main():
    parser = argparse.ArgumentParser(description="HSI Daily Prediction")
    parser.add_argument("--format", choices=["json", "telegram", "both"], default="both",
                       help="Output format")
    parser.add_argument("--save", action="store_true", help="Save prediction to file")
    args = parser.parse_args()
    
    result = predict_today()
    
    if args.format in ["json", "both"]:
        print("\n=== JSON Output ===")
        print(json.dumps(result, indent=2))
    
    if args.format in ["telegram", "both"]:
        print("\n=== Telegram Message ===")
        print(format_telegram(result))
    
    if args.save:
        # Save to predictions log
        log_file = PROJECT_DIR / "data" / "predictions.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"\nSaved to: {log_file}")
    
    return result


if __name__ == "__main__":
    main()
