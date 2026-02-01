#!/usr/bin/env python3
"""
Walk-forward training for HSI forecasting models.
Trains 3 XGBoost models: range_high, range_low, direction
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, roc_auc_score
from pathlib import Path
import json
from datetime import datetime

PROJECT_DIR = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed features and targets."""
    features = pd.read_csv(PROCESSED_DIR / "features.csv", index_col=0, parse_dates=True)
    targets = pd.read_csv(PROCESSED_DIR / "targets.csv", index_col=0, parse_dates=True)
    return features, targets


def walk_forward_split(df: pd.DataFrame, initial_train_months: int = 18, test_months: int = 1):
    """
    Generate walk-forward train/test splits.
    
    Yields:
        (train_idx, test_idx, fold_num)
    """
    dates = df.index.to_series()
    min_date = dates.min()
    max_date = dates.max()
    
    # Initial training end date
    train_end = min_date + pd.DateOffset(months=initial_train_months)
    fold = 1
    
    while train_end + pd.DateOffset(months=test_months) <= max_date:
        test_end = train_end + pd.DateOffset(months=test_months)
        
        train_idx = dates[dates < train_end].index
        test_idx = dates[(dates >= train_end) & (dates < test_end)].index
        
        if len(test_idx) > 0:
            yield train_idx, test_idx, fold
        
        train_end = test_end
        fold += 1


def train_regressor(X_train, y_train, X_test, y_test, name: str) -> tuple[xgb.XGBRegressor, dict]:
    """Train and evaluate a regression model."""
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
    }
    
    return model, metrics


def train_classifier(X_train, y_train, X_test, y_test, name: str) -> tuple[xgb.XGBClassifier, dict]:
    """Train and evaluate a classification model."""
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5,
    }
    
    return model, metrics


def train_walk_forward():
    """
    Perform walk-forward training and evaluation.
    Returns final models trained on all data and evaluation results.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    features, targets = load_data()
    
    # Feature columns (exclude target-related columns)
    feature_cols = features.columns.tolist()
    
    # Store results for each fold
    results = {
        "high": {"mae": [], "rmse": []},
        "low": {"mae": [], "rmse": []},
        "direction": {"accuracy": [], "auc_roc": []}
    }
    
    print("=" * 60)
    print("Walk-Forward Training")
    print("=" * 60)
    
    for train_idx, test_idx, fold in walk_forward_split(features):
        X_train = features.loc[train_idx, feature_cols]
        X_test = features.loc[test_idx, feature_cols]
        
        print(f"\nFold {fold}: Train {len(train_idx)} samples, Test {len(test_idx)} samples")
        print(f"  Train: {train_idx[0].date()} to {train_idx[-1].date()}")
        print(f"  Test:  {test_idx[0].date()} to {test_idx[-1].date()}")
        
        # Train High model
        y_train_high = targets.loc[train_idx, "high_pct"]
        y_test_high = targets.loc[test_idx, "high_pct"]
        _, metrics_high = train_regressor(X_train, y_train_high, X_test, y_test_high, "high")
        results["high"]["mae"].append(metrics_high["mae"])
        results["high"]["rmse"].append(metrics_high["rmse"])
        
        # Train Low model
        y_train_low = targets.loc[train_idx, "low_pct"]
        y_test_low = targets.loc[test_idx, "low_pct"]
        _, metrics_low = train_regressor(X_train, y_train_low, X_test, y_test_low, "low")
        results["low"]["mae"].append(metrics_low["mae"])
        results["low"]["rmse"].append(metrics_low["rmse"])
        
        # Train Direction model
        y_train_dir = targets.loc[train_idx, "direction"]
        y_test_dir = targets.loc[test_idx, "direction"]
        _, metrics_dir = train_classifier(X_train, y_train_dir, X_test, y_test_dir, "direction")
        results["direction"]["accuracy"].append(metrics_dir["accuracy"])
        results["direction"]["auc_roc"].append(metrics_dir["auc_roc"])
        
        print(f"  High:  MAE={metrics_high['mae']:.4f}%, RMSE={metrics_high['rmse']:.4f}%")
        print(f"  Low:   MAE={metrics_low['mae']:.4f}%, RMSE={metrics_low['rmse']:.4f}%")
        print(f"  Dir:   Acc={metrics_dir['accuracy']:.2%}, AUC={metrics_dir['auc_roc']:.4f}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("Walk-Forward Summary (mean ± std)")
    print("=" * 60)
    
    summary = {}
    for model_name, metrics in results.items():
        summary[model_name] = {}
        for metric_name, values in metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            summary[model_name][metric_name] = {"mean": mean_val, "std": std_val}
            print(f"{model_name:10} {metric_name:10}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Train final models on ALL data
    print("\n" + "=" * 60)
    print("Training Final Models on Full Dataset")
    print("=" * 60)
    
    X_all = features[feature_cols]
    
    # Load optimized params if available
    best_params_path = MODELS_DIR / "best_params.json"
    if best_params_path.exists():
        with open(best_params_path) as f:
            best_params = json.load(f)
        print(f"Using optimized params: {best_params}")
    else:
        best_params = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}
        print(f"Using default params: {best_params}")
    
    # Final High model
    model_high = xgb.XGBRegressor(**best_params, random_state=42)
    model_high.fit(X_all, targets["high_pct"])
    model_high.save_model(MODELS_DIR / "range_high.json")
    print(f"Saved: {MODELS_DIR / 'range_high.json'}")
    
    # Final Low model
    model_low = xgb.XGBRegressor(**best_params, random_state=42)
    model_low.fit(X_all, targets["low_pct"])
    model_low.save_model(MODELS_DIR / "range_low.json")
    print(f"Saved: {MODELS_DIR / 'range_low.json'}")
    
    # Final Direction model
    model_dir = xgb.XGBClassifier(**best_params, random_state=42, eval_metric="logloss")
    model_dir.fit(X_all, targets["direction"])
    model_dir.save_model(MODELS_DIR / "direction.json")
    print(f"Saved: {MODELS_DIR / 'direction.json'}")
    
    # Save feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance_high": model_high.feature_importances_,
        "importance_low": model_low.feature_importances_,
        "importance_dir": model_dir.feature_importances_,
    }).sort_values("importance_high", ascending=False)
    
    importance.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
    print(f"Saved: {MODELS_DIR / 'feature_importance.csv'}")
    
    print("\nFeature Importance (High model):")
    print(importance[["feature", "importance_high"]].to_string(index=False))
    
    # Save training metadata
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "n_samples": len(features),
        "n_folds": len(results["high"]["mae"]),
        "feature_cols": feature_cols,
        "summary": summary,
    }
    
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return summary


if __name__ == "__main__":
    train_walk_forward()
