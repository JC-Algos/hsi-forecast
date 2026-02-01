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


def calculate_signal_alignment(features: pd.Series, predicted_direction: str) -> dict:
    """
    Calculate confidence based on how ALL factors align with the predicted direction.
    
    Factors considered:
    - Technical: Price vs EMA20, Price vs MA5, HSI momentum
    - Overnight: FXI, SPX, NDX changes
    - Currency: USDCNH (inverted - CNH strength = bullish)
    - Volatility: VIX level and change
    
    Returns:
        dict with alignment score, factor breakdown, and confidence
    """
    
    # Define all factors with their bullish/bearish interpretation
    # Positive value = bullish signal, negative = bearish
    factors = []
    
    # === TECHNICAL FACTORS ===
    hsi_close = features.get("hsi_prev_close", 0)
    ema20 = features.get("hsi_ema20", hsi_close)
    ma5 = features.get("hsi_ma5", hsi_close)
    hsi_change = features.get("hsi_change_pct", 0)
    
    # Price vs EMA20 (above = bullish)
    if ema20 > 0:
        ema20_signal = (hsi_close - ema20) / ema20
        factors.append({
            "name": "EMA20",
            "signal": ema20_signal,
            "weight": 0.12,
            "display": f"{'>' if hsi_close > ema20 else '<'}EMA20",
            "bullish": hsi_close > ema20
        })
    
    # Price vs MA5 (above = bullish)
    if ma5 > 0:
        ma5_signal = (hsi_close - ma5) / ma5
        factors.append({
            "name": "MA5",
            "signal": ma5_signal,
            "weight": 0.08,
            "display": f"{'>' if hsi_close > ma5 else '<'}MA5",
            "bullish": hsi_close > ma5
        })
    
    # HSI momentum (yesterday's change)
    factors.append({
        "name": "Momentum",
        "signal": hsi_change,
        "weight": 0.08,
        "display": f"HSI{hsi_change*100:+.1f}%",
        "bullish": hsi_change > 0
    })
    
    # === OVERNIGHT FACTORS ===
    fxi_change = features.get("fxi_change_pct", 0)
    spx_change = features.get("spx_change_pct", 0)
    ndx_change = features.get("ndx_change_pct", 0)
    
    # FXI (most important overnight indicator)
    factors.append({
        "name": "FXI",
        "signal": fxi_change,
        "weight": 0.20,  # Highest weight - best HK proxy
        "display": f"FXI{fxi_change*100:+.1f}%",
        "bullish": fxi_change > 0
    })
    
    # SPX
    factors.append({
        "name": "SPX",
        "signal": spx_change,
        "weight": 0.12,
        "display": f"SPX{spx_change*100:+.1f}%",
        "bullish": spx_change > 0
    })
    
    # NDX
    factors.append({
        "name": "NDX",
        "signal": ndx_change,
        "weight": 0.10,
        "display": f"NDX{ndx_change*100:+.1f}%",
        "bullish": ndx_change > 0
    })
    
    # === CURRENCY FACTOR ===
    usdcnh_change = features.get("usdcnh_change_pct", 0)
    # USDCNH up = USD stronger = CNH weaker = bearish for HK
    factors.append({
        "name": "CNH",
        "signal": -usdcnh_change,  # Inverted
        "weight": 0.10,
        "display": f"CNH{-usdcnh_change*100:+.1f}%",
        "bullish": usdcnh_change < 0  # CNH strengthening is bullish
    })
    
    # === VOLATILITY FACTORS ===
    vix = features.get("vix", 20)
    vix_change = features.get("vix_change_pct", 0)
    
    # VIX level (low VIX = bullish, high = bearish)
    # Neutral at 20, bullish below, bearish above
    vix_level_signal = (20 - vix) / 20  # Positive when VIX < 20
    factors.append({
        "name": "VIX_Level",
        "signal": vix_level_signal,
        "weight": 0.10,
        "display": f"VIX={vix:.0f}",
        "bullish": vix < 20
    })
    
    # VIX change (VIX down = bullish, up = bearish)
    factors.append({
        "name": "VIX_Chg",
        "signal": -vix_change,  # Inverted
        "weight": 0.10,
        "display": f"VIX{vix_change*100:+.1f}%",
        "bullish": vix_change < 0
    })
    
    # === CALCULATE DIFFUSION ===
    # Diffusion = bullish count - bearish count
    # Positive = net bullish, Negative = net bearish
    
    bullish_factors = []
    bearish_factors = []
    neutral_factors = []
    
    # Also calculate weighted diffusion
    weighted_bull = 0.0
    weighted_bear = 0.0
    total_weight = 0.0
    
    for f in factors:
        total_weight += f["weight"]
        threshold = 0.001  # 0.1% threshold
        
        if abs(f["signal"]) < threshold:
            neutral_factors.append(f)
        elif f["bullish"]:
            bullish_factors.append(f)
            weighted_bull += f["weight"]
        else:
            bearish_factors.append(f)
            weighted_bear += f["weight"]
    
    # Simple diffusion (count-based)
    diffusion = len(bullish_factors) - len(bearish_factors)
    
    # Weighted diffusion (-1 to +1 scale)
    weighted_diffusion = (weighted_bull - weighted_bear) / total_weight if total_weight > 0 else 0
    
    # Diffusion direction
    if diffusion > 0:
        diffusion_direction = "UP"
    elif diffusion < 0:
        diffusion_direction = "DOWN"
    else:
        diffusion_direction = "NEUTRAL"
    
    # Signal strength (average magnitude of all non-neutral factors)
    active_factors = bullish_factors + bearish_factors
    if active_factors:
        avg_magnitude = sum(abs(f["signal"]) for f in active_factors) / len(active_factors)
        strength = min(1.0, avg_magnitude / 0.01)  # 1% avg = full strength
    else:
        strength = 0.0
    
    # Format display
    bull_display = [f["display"] for f in bullish_factors]
    bear_display = [f["display"] for f in bearish_factors]
    
    return {
        "diffusion": diffusion,  # Simple count: bull - bear
        "weighted_diffusion": weighted_diffusion,  # Weighted: -1 to +1
        "diffusion_direction": diffusion_direction,
        "strength": strength,
        "bullish_count": len(bullish_factors),
        "bearish_count": len(bearish_factors),
        "neutral_count": len(neutral_factors),
        "total_factors": len(factors),
        "bullish_factors": bull_display,
        "bearish_factors": bear_display,
    }


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
    predicted_direction = "UP" if direction_prob > 0.5 else "DOWN"
    
    # Calculate signal alignment for confidence
    signal_alignment = calculate_signal_alignment(features, predicted_direction)
    
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
    
    # === DIFFUSION-BASED CONFIDENCE ===
    # Diffusion = bullish - bearish (range: -9 to +9)
    # Positive = net bullish, Negative = net bearish
    diffusion = signal_alignment["diffusion"]
    total_factors = signal_alignment["total_factors"]
    
    # Check if model and diffusion agree
    model_is_bullish = (predicted_direction == "UP")
    diffusion_is_bullish = (diffusion > 0)
    diffusion_is_bearish = (diffusion < 0)
    
    # Agreement check:
    # - Model UP + Diffusion positive = agree
    # - Model DOWN + Diffusion negative = agree
    # - Otherwise = disagree (or neutral if diffusion = 0)
    
    if diffusion == 0:
        # Neutral - no boost or penalty
        agrees = True  # Treat as neutral agreement
        final_confidence = 0.50
    elif (model_is_bullish and diffusion_is_bullish) or (not model_is_bullish and diffusion_is_bearish):
        # Model and diffusion AGREE
        agrees = True
        # Confidence = 50% + (|diffusion|/9 × 50%)
        final_confidence = 0.50 + (abs(diffusion) / total_factors) * 0.50
    else:
        # Model and diffusion DISAGREE
        agrees = False
        # Confidence = 50% - (|diffusion|/9 × 50%)
        final_confidence = 0.50 - (abs(diffusion) / total_factors) * 0.50
    
    result = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "prediction_time": datetime.now().isoformat(),
        "reference_price": float(yesterday_close),
        "predicted_high_pct": float(round(high_pct, 3)),
        "predicted_low_pct": float(round(low_pct, 3)),
        "predicted_high": int(round(predicted_high)),
        "predicted_low": int(round(predicted_low)),
        "predicted_range": int(round(predicted_high - predicted_low)),
        "predicted_range_pct": float(round(high_pct - low_pct, 3)),
        "model_prob": float(round(direction_prob, 3)),
        "direction": predicted_direction,
        "confidence": float(round(final_confidence, 3)),
        "diffusion": diffusion,
        "bullish_count": signal_alignment["bullish_count"],
        "bearish_count": signal_alignment["bearish_count"],
        "total_factors": total_factors,
        "bullish_factors": signal_alignment["bullish_factors"],
        "bearish_factors": signal_alignment["bearish_factors"],
        "model_diffusion_agree": agrees,
        "volatility_regime": vol_regime,
        "volatility_multiplier": float(round(vol_multiplier, 2)),
        "gap_adjustment": gap_reason,
    }
    
    return result


def format_telegram(result: dict) -> str:
    """Format prediction for Telegram message."""
    from datetime import datetime, timedelta
    
    # Calculate next trading day (skip weekends)
    today = datetime.now()
    if today.weekday() == 5:  # Saturday
        next_trading = today + timedelta(days=2)
    elif today.weekday() == 6:  # Sunday
        next_trading = today + timedelta(days=1)
    elif today.weekday() == 4:  # Friday
        next_trading = today + timedelta(days=3)
    else:
        next_trading = today + timedelta(days=1)
    
    day_name = next_trading.strftime("%A")
    date_str = next_trading.strftime("%-d %b %Y")
    
    # Reference day (previous trading day)
    ref_day = "Fri" if today.weekday() in [5, 6, 0] else today.strftime("%a")
    
    direction_emoji = "📈" if result["direction"] == "UP" else "📉"
    confidence_pct = result["confidence"]
    confidence_bar = "█" * int(confidence_pct * 10) + "░" * (10 - int(confidence_pct * 10))
    
    # Diffusion info
    diffusion = result.get("diffusion", 0)
    bullish_count = result.get("bullish_count", 0)
    bearish_count = result.get("bearish_count", 0)
    agree = result.get("model_diffusion_agree", True)
    
    bullish_factors = result.get("bullish_factors", [])
    bearish_factors = result.get("bearish_factors", [])
    
    # Format factor lists
    bull_str = ", ".join(bullish_factors) if bullish_factors else "None"
    bear_str = ", ".join(bearish_factors) if bearish_factors else "None"
    
    # Model probability (convert to direction %)
    model_prob = result.get("model_prob", 0.5)
    if result["direction"] == "DOWN":
        direction_prob = (1 - model_prob) * 100
    else:
        direction_prob = model_prob * 100
    
    msg = f"""🎯 HSI Daily Forecast - {day_name} {date_str}

📊 Predicted Range (from {ref_day} close {result['reference_price']:,.0f}):
• High: {result['predicted_high']:,} (+{result['predicted_high_pct']:.2f}%)
• Low: {result['predicted_low']:,} ({result['predicted_low_pct']:.2f}%)
• Range: {result['predicted_range']:,} pts ({result['predicted_range_pct']:.2f}%)

{direction_emoji} Direction: {result['direction']} ({direction_prob:.0f}% probability)
Confidence: [{confidence_bar}] {confidence_pct:.0%}
Diffusion: {diffusion:+d} ({bullish_count}🟢 vs {bearish_count}🔴) {"✓" if agree else "⚠️"}

🟢 Bullish: {bull_str}
🔴 Bearish: {bear_str}

⚡ Volatility Regime: {result['volatility_regime']} (×{result['volatility_multiplier']:.1f})

Model: XGBoost | Accuracy: 72% | MAE: 0.5%
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
