#!/usr/bin/env python3
"""
Daily prediction for HSI range and direction.
Run before market open to get today's forecast.

Supports two modes:
- legacy: XGBoost only (original)
- hybrid: Ensemble (XGB+LGB+CAT) for High/Low, CatBoost only for Direction
"""

import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import json
import argparse

PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = PROJECT_DIR / "models"
ENSEMBLE_DIR = MODELS_DIR / "ensemble"

# Default to hybrid mode (ensemble for high/low, catboost for direction)
USE_HYBRID = True


class GRUModel(nn.Module):
    """GRU model for direction prediction."""
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out, _ = self.gru(x)
        return self.sigmoid(self.fc(out[:, -1, :]))


def load_gru_model():
    """Load trained GRU model and scaler."""
    gru_path = MODELS_DIR / "gru_direction.pt"
    scaler_path = MODELS_DIR / "gru_scaler.pkl"
    
    if not gru_path.exists() or not scaler_path.exists():
        return None, None, None
    
    checkpoint = torch.load(gru_path, weights_only=False)
    hp = checkpoint['hyperparams']
    
    model = GRUModel(hp['input_size'], hp['hidden_size'], hp['num_layers'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler = joblib.load(scaler_path)
    
    return model, scaler, checkpoint['feature_cols']


def predict_direction_arima(df: pd.DataFrame) -> tuple[float, str, bool]:
    """
    Use ARIMA model to predict direction.
    
    Returns:
        (prob, direction, success): probability, direction string, whether prediction succeeded
    """
    from statsmodels.tsa.arima.model import ARIMA
    import warnings
    warnings.filterwarnings('ignore')
    
    arima_path = MODELS_DIR / "arima_params.pkl"
    if not arima_path.exists():
        return 0.0, "N/A", False
    
    arima_params = joblib.load(arima_path)
    order = arima_params['order']
    
    # Use processed features for consistent returns data
    processed_dir = PROJECT_DIR / "data" / "processed"
    features_df = pd.read_csv(processed_dir / "features.csv", index_col=0, parse_dates=True)
    
    if 'hsi_change_pct' not in features_df.columns:
        return 0.0, "N/A", False
    
    returns = features_df['hsi_change_pct'].dropna()
    
    if len(returns) < 50:
        return 0.0, "N/A", False
    
    try:
        model = ARIMA(returns, order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=1)
        forecast_val = float(forecast.values[0])
        
        direction = "UP" if forecast_val > 0 else "DOWN"
        # Convert forecast to pseudo-probability (sigmoid-like)
        prob = 1 / (1 + np.exp(-forecast_val * 100))  # Scale for reasonable range
        
        return prob, direction, True
    except Exception as e:
        return 0.0, "N/A", False


class EnsembleRegressor:
    """Ensemble of XGBoost, LightGBM, CatBoost for regression."""
    
    def __init__(self, name: str):
        self.name = name
        self.models = {}
        self.weights = {"xgb": 0.33, "lgb": 0.33, "cat": 0.34}
    
    def load(self, path: Path):
        """Load all models and weights."""
        self.models["xgb"] = xgb.XGBRegressor()
        self.models["xgb"].load_model(path / f"{self.name}_xgb.json")
        
        self.models["lgb"] = joblib.load(path / f"{self.name}_lgb.pkl")
        
        self.models["cat"] = cb.CatBoostRegressor()
        self.models["cat"].load_model(path / f"{self.name}_cat.cbm")
        
        with open(path / f"{self.name}_weights.json") as f:
            self.weights = json.load(f)
    
    def predict(self, X):
        """Weighted average prediction."""
        preds = {
            "xgb": self.models["xgb"].predict(X),
            "lgb": self.models["lgb"].predict(X),
            "cat": self.models["cat"].predict(X),
        }
        
        return (
            self.weights["xgb"] * preds["xgb"] +
            self.weights["lgb"] * preds["lgb"] +
            self.weights["cat"] * preds["cat"]
        )


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


def load_models(hybrid: bool = True):
    """
    Load trained models.
    
    Args:
        hybrid: If True, use XGBoost for Range + CatBoost for Direction (best combo)
                If False, use XGBoost only (legacy mode)
    """
    if hybrid and ENSEMBLE_DIR.exists():
        # Best combo: XGBoost for Range (lower MAE) + CatBoost for Direction (71% acc)
        model_high = xgb.XGBRegressor()
        model_high.load_model(MODELS_DIR / "range_high.json")
        
        model_low = xgb.XGBRegressor()
        model_low.load_model(MODELS_DIR / "range_low.json")
        
        # CatBoost for direction (71.4% vs XGBoost 47.6%)
        model_dir = cb.CatBoostClassifier()
        model_dir.load_model(ENSEMBLE_DIR / "direction_cat.cbm")
        
        return model_high, model_low, model_dir, "xgb+catboost"
    else:
        # Legacy mode: XGBoost only
        model_high = xgb.XGBRegressor()
        model_high.load_model(MODELS_DIR / "range_high.json")
        
        model_low = xgb.XGBRegressor()
        model_low.load_model(MODELS_DIR / "range_low.json")
        
        model_dir = xgb.XGBClassifier()
        model_dir.load_model(MODELS_DIR / "direction.json")
        
        return model_high, model_low, model_dir, "xgboost"


def predict_direction_gru(df: pd.DataFrame, features: pd.Series) -> tuple[float, bool]:
    """
    Use GRU model to predict direction.
    
    Returns:
        (probability, success): probability of UP direction and whether prediction succeeded
    """
    gru_model, gru_scaler, gru_feature_cols = load_gru_model()
    
    if gru_model is None:
        return 0.5, False
    
    # Need sequence of last 10 days
    SEQUENCE_LENGTH = 10
    
    # Get processed features from the data
    processed_dir = PROJECT_DIR / "data" / "processed"
    features_df = pd.read_csv(processed_dir / "features.csv", index_col=0, parse_dates=True)
    
    if len(features_df) < SEQUENCE_LENGTH:
        return 0.5, False
    
    # Get last SEQUENCE_LENGTH rows
    X_recent = features_df[gru_feature_cols].iloc[-SEQUENCE_LENGTH:].values
    
    # Scale
    X_scaled = gru_scaler.transform(X_recent)
    
    # Create sequence tensor
    X_seq = torch.FloatTensor(X_scaled).unsqueeze(0)  # (1, seq_len, features)
    
    # Predict
    with torch.no_grad():
        prob = gru_model(X_seq).item()
    
    return prob, True


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


def predict_today(output_format: str = "json", use_hybrid: bool = True) -> dict:
    """
    Make prediction for today's HSI range and direction.
    
    Args:
        output_format: Output format (json/telegram)
        use_hybrid: If True, use Ensemble for High/Low and CatBoost for Direction
    
    Returns:
        dict with prediction results
    """
    # Fetch data and calculate features
    df = fetch_recent_data(days=30)
    features = calculate_features(df)
    
    # Load models (hybrid or legacy)
    model_high, model_low, model_dir, model_type = load_models(hybrid=use_hybrid)
    
    # Get feature order from metadata
    metadata_path = ENSEMBLE_DIR / "metadata.json" if model_type == "hybrid" else MODELS_DIR / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    feature_cols = metadata["feature_cols"]
    
    # Prepare input
    X = features[feature_cols].values.reshape(1, -1)
    
    # Predict base values
    high_pct_base = model_high.predict(X)[0]
    low_pct_base = model_low.predict(X)[0]
    
    # Direction prediction (CatBoost in hybrid mode, XGBoost in legacy)
    direction_prob = model_dir.predict_proba(X)[0, 1]
    predicted_direction = "UP" if direction_prob > 0.5 else "DOWN"
    
    # GRU direction prediction (if available)
    gru_prob, gru_success = predict_direction_gru(df, features)
    gru_direction = "UP" if gru_prob > 0.5 else "DOWN" if gru_success else None
    
    # ARIMA direction prediction
    arima_prob, arima_direction, arima_success = predict_direction_arima(df)
    
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
        "model_type": model_type,  # "hybrid" or "xgboost"
        # GRU model predictions
        "gru_prob": float(round(gru_prob, 3)) if gru_success else None,
        "gru_direction": gru_direction,
        "gru_available": gru_success,
        # ARIMA model predictions
        "arima_prob": float(round(arima_prob, 3)) if arima_success else None,
        "arima_direction": arima_direction,
        "arima_available": arima_success,
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
    
    # Diffusion info
    diffusion = result.get("diffusion", 0)
    bullish_count = result.get("bullish_count", 0)
    bearish_count = result.get("bearish_count", 0)
    diffusion_direction = "UP" if diffusion > 0 else "DOWN" if diffusion < 0 else "NEUTRAL"
    
    bullish_factors = result.get("bullish_factors", [])
    bearish_factors = result.get("bearish_factors", [])
    
    # Format factor lists
    bull_str = ", ".join(bullish_factors) if bullish_factors else "None"
    bear_str = ", ".join(bearish_factors) if bearish_factors else "None"
    
    # === 4 Direction Models ===
    # 1. CatBoost (ML)
    catboost_dir = result["direction"]
    catboost_prob = result.get("model_prob", 0.5)
    catboost_pct = catboost_prob * 100 if catboost_dir == "UP" else (1 - catboost_prob) * 100
    catboost_emoji = "📈" if catboost_dir == "UP" else "📉"
    
    # 2. GRU (Deep Learning)
    gru_available = result.get('gru_available', False)
    if gru_available:
        gru_dir = result['gru_direction']
        gru_prob = result['gru_prob']
        gru_pct = gru_prob * 100 if gru_dir == "UP" else (1 - gru_prob) * 100
        gru_emoji = "📈" if gru_dir == "UP" else "📉"
    else:
        gru_dir = None
        gru_pct = 0
        gru_emoji = "❓"
    
    # 3. ARIMA (Statistical)
    arima_available = result.get('arima_available', False)
    if arima_available:
        arima_dir = result['arima_direction']
        arima_prob = result['arima_prob']
        arima_pct = arima_prob * 100 if arima_dir == "UP" else (1 - arima_prob) * 100
        arima_emoji = "📈" if arima_dir == "UP" else "📉"
    else:
        arima_dir = None
        arima_pct = 0
        arima_emoji = "❓"
    
    # 4. Diffusion (Rule-based)
    diff_emoji = "📈" if diffusion_direction == "UP" else "📉" if diffusion_direction == "DOWN" else "➖"
    
    # Count agreements (4 judges)
    directions = [catboost_dir]
    if gru_available:
        directions.append(gru_dir)
    if arima_available:
        directions.append(arima_dir)
    directions.append(diffusion_direction)
    
    up_count = directions.count("UP")
    down_count = directions.count("DOWN")
    total_judges = len([d for d in directions if d in ["UP", "DOWN"]])
    
    # Consensus with 3:1 confidence
    if up_count >= 3 and down_count <= 1:
        consensus = "UP"
        consensus_emoji = "📈"
        confidence = "✅ 強烈" if up_count == 4 else "✅ 有信心"
    elif down_count >= 3 and up_count <= 1:
        consensus = "DOWN"
        consensus_emoji = "📉"
        confidence = "✅ 強烈" if down_count == 4 else "✅ 有信心"
    elif up_count > down_count:
        consensus = "UP"
        consensus_emoji = "📈"
        confidence = "⚠️ 弱"
    elif down_count > up_count:
        consensus = "DOWN"
        consensus_emoji = "📉"
        confidence = "⚠️ 弱"
    else:
        consensus = "MIXED"
        consensus_emoji = "⚖️"
        confidence = "❌ 無方向"
    
    vote_str = f"{up_count}↑ vs {down_count}↓"
    
    msg = f"""🎯 HSI Daily Forecast - {day_name} {date_str}

📊 Predicted Range (from {ref_day} close {result['reference_price']:,.0f}):
• High: {result['predicted_high']:,} (+{result['predicted_high_pct']:.2f}%)
• Low: {result['predicted_low']:,} ({result['predicted_low_pct']:.2f}%)
• Range: {result['predicted_range']:,} pts ({result['predicted_range_pct']:.2f}%)

{consensus_emoji} Direction: {consensus} ({vote_str}) | {confidence}

🗳️ 4 Judges:
• CatBoost (ML): {catboost_emoji} {catboost_dir} ({catboost_pct:.0f}%)
• GRU (Deep): {gru_emoji} {gru_dir if gru_available else 'N/A'}{f' ({gru_pct:.0f}%)' if gru_available else ''}
• ARIMA (Stats): {arima_emoji} {arima_dir if arima_available else 'N/A'}{f' ({arima_pct:.0f}%)' if arima_available else ''}
• Diffusion: {diff_emoji} {diffusion_direction} ({bullish_count}🟢 vs {bearish_count}🔴)

🟢 Bullish: {bull_str}
🔴 Bearish: {bear_str}

⚡ Volatility: {result['volatility_regime']} (×{result['volatility_multiplier']:.1f})
"""
    return msg


def main():
    parser = argparse.ArgumentParser(description="HSI Daily Prediction")
    parser.add_argument("--format", choices=["json", "telegram", "both"], default="both",
                       help="Output format")
    parser.add_argument("--save", action="store_true", help="Save prediction to file")
    parser.add_argument("--legacy", action="store_true", help="Use legacy XGBoost-only mode")
    args = parser.parse_args()
    
    result = predict_today(use_hybrid=not args.legacy)
    
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
