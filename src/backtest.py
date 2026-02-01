#!/usr/bin/env python3
"""
Backtest and performance analysis for HSI forecasting models.
Evaluates model performance on historical data.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, roc_auc_score, confusion_matrix
from pathlib import Path
import json
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"


def load_data():
    """Load processed features and targets."""
    features = pd.read_csv(PROCESSED_DIR / "features.csv", index_col=0, parse_dates=True)
    targets = pd.read_csv(PROCESSED_DIR / "targets.csv", index_col=0, parse_dates=True)
    return features, targets


def load_models():
    """Load trained models."""
    model_high = xgb.XGBRegressor()
    model_high.load_model(MODELS_DIR / "range_high.json")
    
    model_low = xgb.XGBRegressor()
    model_low.load_model(MODELS_DIR / "range_low.json")
    
    model_dir = xgb.XGBClassifier()
    model_dir.load_model(MODELS_DIR / "direction.json")
    
    with open(MODELS_DIR / "metadata.json") as f:
        metadata = json.load(f)
    
    return model_high, model_low, model_dir, metadata


def evaluate_range_model(y_true, y_pred, name: str) -> dict:
    """Evaluate range prediction model."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Percentage within threshold
    errors = np.abs(y_true - y_pred)
    within_0_5 = (errors <= 0.5).mean()  # Within 0.5%
    within_1_0 = (errors <= 1.0).mean()  # Within 1.0%
    
    print(f"\n{name} Model Performance:")
    print(f"  MAE:  {mae:.4f}%")
    print(f"  RMSE: {rmse:.4f}%")
    print(f"  Within ±0.5%: {within_0_5:.1%}")
    print(f"  Within ±1.0%: {within_1_0:.1%}")
    
    return {
        "mae": mae,
        "rmse": rmse,
        "within_0.5%": within_0_5,
        "within_1.0%": within_1_0,
    }


def evaluate_direction_model(y_true, y_pred, y_prob) -> dict:
    """Evaluate direction prediction model."""
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\nDirection Model Performance:")
    print(f"  Accuracy:  {acc:.1%}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn}, FP={fp}")
    print(f"  FN={fn}, TP={tp}")
    
    return {
        "accuracy": acc,
        "auc_roc": auc,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def analyze_predictions(features, targets, model_high, model_low, model_dir, feature_cols) -> pd.DataFrame:
    """Generate predictions for all data and analyze."""
    X = features[feature_cols]
    
    predictions = pd.DataFrame(index=features.index)
    predictions["actual_high_pct"] = targets["high_pct"]
    predictions["actual_low_pct"] = targets["low_pct"]
    predictions["actual_direction"] = targets["direction"]
    
    predictions["pred_high_pct"] = model_high.predict(X)
    predictions["pred_low_pct"] = model_low.predict(X)
    predictions["pred_direction_prob"] = model_dir.predict_proba(X)[:, 1]
    predictions["pred_direction"] = (predictions["pred_direction_prob"] > 0.5).astype(int)
    
    # Calculate range capture rate
    # Did the predicted range contain the actual high/low?
    predictions["high_error"] = predictions["actual_high_pct"] - predictions["pred_high_pct"]
    predictions["low_error"] = predictions["actual_low_pct"] - predictions["pred_low_pct"]
    
    return predictions


def run_backtest():
    """Run full backtest analysis."""
    print("=" * 60)
    print("HSI Forecast Model Backtest")
    print("=" * 60)
    
    # Load data and models
    features, targets = load_data()
    model_high, model_low, model_dir, metadata = load_models()
    feature_cols = metadata["feature_cols"]
    
    print(f"\nDataset: {len(features)} samples")
    print(f"Date range: {features.index[0].date()} to {features.index[-1].date()}")
    
    # Generate predictions
    predictions = analyze_predictions(features, targets, model_high, model_low, model_dir, feature_cols)
    
    # Evaluate each model
    print("\n" + "-" * 40)
    high_metrics = evaluate_range_model(
        predictions["actual_high_pct"],
        predictions["pred_high_pct"],
        "Range High"
    )
    
    low_metrics = evaluate_range_model(
        predictions["actual_low_pct"],
        predictions["pred_low_pct"],
        "Range Low"
    )
    
    dir_metrics = evaluate_direction_model(
        predictions["actual_direction"],
        predictions["pred_direction"],
        predictions["pred_direction_prob"]
    )
    
    # Analyze by confidence level
    print("\n" + "-" * 40)
    print("\nDirection Accuracy by Confidence Level:")
    predictions["confidence"] = np.abs(predictions["pred_direction_prob"] - 0.5) * 2
    
    for threshold in [0.3, 0.5, 0.7]:
        high_conf = predictions[predictions["confidence"] >= threshold]
        if len(high_conf) > 0:
            acc = (high_conf["actual_direction"] == high_conf["pred_direction"]).mean()
            print(f"  Confidence >= {threshold:.0%}: {acc:.1%} accuracy ({len(high_conf)} samples)")
    
    # Save predictions
    predictions.to_csv(PROCESSED_DIR / "backtest_predictions.csv")
    print(f"\nPredictions saved to: {PROCESSED_DIR / 'backtest_predictions.csv'}")
    
    # Save metrics
    all_metrics = {
        "range_high": high_metrics,
        "range_low": low_metrics,
        "direction": dir_metrics,
    }
    
    with open(MODELS_DIR / "backtest_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"Metrics saved to: {MODELS_DIR / 'backtest_metrics.json'}")
    
    return all_metrics, predictions


if __name__ == "__main__":
    metrics, predictions = run_backtest()
