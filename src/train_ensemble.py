#!/usr/bin/env python3
"""
Ensemble training for HSI forecasting: XGBoost + LightGBM + CatBoost
Uses walk-forward validation to determine optimal weights.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, roc_auc_score
from sklearn.linear_model import Ridge
from pathlib import Path
import json
from datetime import datetime
import joblib

PROJECT_DIR = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"
ENSEMBLE_DIR = MODELS_DIR / "ensemble"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed features and targets."""
    features = pd.read_csv(PROCESSED_DIR / "features.csv", index_col=0, parse_dates=True)
    targets = pd.read_csv(PROCESSED_DIR / "targets.csv", index_col=0, parse_dates=True)
    return features, targets


def walk_forward_split(df: pd.DataFrame, initial_train_months: int = 18, test_months: int = 1):
    """Generate walk-forward train/test splits."""
    dates = df.index.to_series()
    min_date = dates.min()
    max_date = dates.max()
    
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


class EnsembleRegressor:
    """Ensemble of XGBoost, LightGBM, CatBoost for regression."""
    
    def __init__(self, name: str):
        self.name = name
        self.models = {}
        self.weights = {"xgb": 0.4, "lgb": 0.35, "cat": 0.25}  # Default weights
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train all three models."""
        
        # XGBoost
        self.models["xgb"] = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.models["xgb"].fit(X_train, y_train)
        
        # LightGBM
        self.models["lgb"] = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        self.models["lgb"].fit(X_train, y_train)
        
        # CatBoost
        self.models["cat"] = cb.CatBoostRegressor(
            iterations=100,
            depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=False
        )
        self.models["cat"].fit(X_train, y_train)
        
        # Optimize weights if validation set provided
        if X_val is not None and y_val is not None:
            self._optimize_weights(X_val, y_val)
    
    def _optimize_weights(self, X_val, y_val):
        """Find optimal weights using validation set."""
        preds = {
            "xgb": self.models["xgb"].predict(X_val),
            "lgb": self.models["lgb"].predict(X_val),
            "cat": self.models["cat"].predict(X_val),
        }
        
        # Grid search for best weights
        best_mae = float("inf")
        best_weights = self.weights.copy()
        
        for w1 in np.arange(0.2, 0.7, 0.05):
            for w2 in np.arange(0.2, 0.7, 0.05):
                w3 = 1 - w1 - w2
                if w3 < 0.1 or w3 > 0.6:
                    continue
                    
                ensemble_pred = w1 * preds["xgb"] + w2 * preds["lgb"] + w3 * preds["cat"]
                mae = mean_absolute_error(y_val, ensemble_pred)
                
                if mae < best_mae:
                    best_mae = mae
                    best_weights = {"xgb": w1, "lgb": w2, "cat": w3}
        
        self.weights = best_weights
    
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
    
    def predict_individual(self, X):
        """Get predictions from each model separately."""
        return {
            "xgb": self.models["xgb"].predict(X),
            "lgb": self.models["lgb"].predict(X),
            "cat": self.models["cat"].predict(X),
            "ensemble": self.predict(X),
        }
    
    def save(self, path: Path):
        """Save all models and weights."""
        path.mkdir(parents=True, exist_ok=True)
        
        self.models["xgb"].save_model(path / f"{self.name}_xgb.json")
        joblib.dump(self.models["lgb"], path / f"{self.name}_lgb.pkl")
        self.models["cat"].save_model(path / f"{self.name}_cat.cbm")
        
        with open(path / f"{self.name}_weights.json", "w") as f:
            json.dump(self.weights, f, indent=2)
    
    def load(self, path: Path):
        """Load all models and weights."""
        self.models["xgb"] = xgb.XGBRegressor()
        self.models["xgb"].load_model(path / f"{self.name}_xgb.json")
        
        self.models["lgb"] = joblib.load(path / f"{self.name}_lgb.pkl")
        
        self.models["cat"] = cb.CatBoostRegressor()
        self.models["cat"].load_model(path / f"{self.name}_cat.cbm")
        
        with open(path / f"{self.name}_weights.json") as f:
            self.weights = json.load(f)


class EnsembleClassifier:
    """Ensemble of XGBoost, LightGBM, CatBoost for classification."""
    
    def __init__(self, name: str):
        self.name = name
        self.models = {}
        self.weights = {"xgb": 0.4, "lgb": 0.35, "cat": 0.25}
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train all three models."""
        
        # XGBoost
        self.models["xgb"] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        )
        self.models["xgb"].fit(X_train, y_train)
        
        # LightGBM
        self.models["lgb"] = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        self.models["lgb"].fit(X_train, y_train)
        
        # CatBoost
        self.models["cat"] = cb.CatBoostClassifier(
            iterations=100,
            depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=False
        )
        self.models["cat"].fit(X_train, y_train)
        
        if X_val is not None and y_val is not None:
            self._optimize_weights(X_val, y_val)
    
    def _optimize_weights(self, X_val, y_val):
        """Find optimal weights using validation set."""
        probs = {
            "xgb": self.models["xgb"].predict_proba(X_val)[:, 1],
            "lgb": self.models["lgb"].predict_proba(X_val)[:, 1],
            "cat": self.models["cat"].predict_proba(X_val)[:, 1],
        }
        
        best_acc = 0
        best_weights = self.weights.copy()
        
        for w1 in np.arange(0.2, 0.7, 0.05):
            for w2 in np.arange(0.2, 0.7, 0.05):
                w3 = 1 - w1 - w2
                if w3 < 0.1 or w3 > 0.6:
                    continue
                    
                ensemble_prob = w1 * probs["xgb"] + w2 * probs["lgb"] + w3 * probs["cat"]
                ensemble_pred = (ensemble_prob > 0.5).astype(int)
                acc = accuracy_score(y_val, ensemble_pred)
                
                if acc > best_acc:
                    best_acc = acc
                    best_weights = {"xgb": w1, "lgb": w2, "cat": w3}
        
        self.weights = best_weights
    
    def predict_proba(self, X):
        """Weighted average probability."""
        probs = {
            "xgb": self.models["xgb"].predict_proba(X)[:, 1],
            "lgb": self.models["lgb"].predict_proba(X)[:, 1],
            "cat": self.models["cat"].predict_proba(X)[:, 1],
        }
        
        return (
            self.weights["xgb"] * probs["xgb"] +
            self.weights["lgb"] * probs["lgb"] +
            self.weights["cat"] * probs["cat"]
        )
    
    def predict(self, X):
        """Predict class labels."""
        return (self.predict_proba(X) > 0.5).astype(int)
    
    def predict_individual(self, X):
        """Get predictions from each model separately."""
        return {
            "xgb": self.models["xgb"].predict(X),
            "lgb": self.models["lgb"].predict(X),
            "cat": self.models["cat"].predict(X),
            "ensemble": self.predict(X),
        }
    
    def save(self, path: Path):
        """Save all models and weights."""
        path.mkdir(parents=True, exist_ok=True)
        
        self.models["xgb"].save_model(path / f"{self.name}_xgb.json")
        joblib.dump(self.models["lgb"], path / f"{self.name}_lgb.pkl")
        self.models["cat"].save_model(path / f"{self.name}_cat.cbm")
        
        with open(path / f"{self.name}_weights.json", "w") as f:
            json.dump(self.weights, f, indent=2)
    
    def load(self, path: Path):
        """Load all models and weights."""
        self.models["xgb"] = xgb.XGBClassifier()
        self.models["xgb"].load_model(path / f"{self.name}_xgb.json")
        
        self.models["lgb"] = joblib.load(path / f"{self.name}_lgb.pkl")
        
        self.models["cat"] = cb.CatBoostClassifier()
        self.models["cat"].load_model(path / f"{self.name}_cat.cbm")
        
        with open(path / f"{self.name}_weights.json") as f:
            self.weights = json.load(f)


def train_ensemble_walk_forward():
    """Perform walk-forward training with ensemble models."""
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    
    features, targets = load_data()
    feature_cols = features.columns.tolist()
    
    # Results storage
    results = {
        "high": {"xgb": [], "lgb": [], "cat": [], "ensemble": []},
        "low": {"xgb": [], "lgb": [], "cat": [], "ensemble": []},
        "direction": {"xgb": [], "lgb": [], "cat": [], "ensemble": []},
    }
    
    print("=" * 70)
    print("Ensemble Walk-Forward Training: XGBoost + LightGBM + CatBoost")
    print("=" * 70)
    
    all_weights = {"high": [], "low": [], "direction": []}
    
    for train_idx, test_idx, fold in walk_forward_split(features):
        X_train = features.loc[train_idx, feature_cols]
        X_test = features.loc[test_idx, feature_cols]
        
        # Use last 20% of train as validation for weight optimization
        val_split = int(len(train_idx) * 0.8)
        X_tr = X_train.iloc[:val_split]
        X_val = X_train.iloc[val_split:]
        
        print(f"\nFold {fold}: Train {len(train_idx)}, Test {len(test_idx)}")
        print(f"  Period: {test_idx[0].date()} to {test_idx[-1].date()}")
        
        # === HIGH ===
        y_train_high = targets.loc[train_idx, "high_pct"]
        y_test_high = targets.loc[test_idx, "high_pct"]
        y_tr_high = y_train_high.iloc[:val_split]
        y_val_high = y_train_high.iloc[val_split:]
        
        model_high = EnsembleRegressor("high")
        model_high.fit(X_tr, y_tr_high, X_val, y_val_high)
        all_weights["high"].append(model_high.weights.copy())
        
        preds_high = model_high.predict_individual(X_test)
        for key, pred in preds_high.items():
            mae = mean_absolute_error(y_test_high, pred)
            results["high"][key].append(mae)
        
        # === LOW ===
        y_train_low = targets.loc[train_idx, "low_pct"]
        y_test_low = targets.loc[test_idx, "low_pct"]
        y_tr_low = y_train_low.iloc[:val_split]
        y_val_low = y_train_low.iloc[val_split:]
        
        model_low = EnsembleRegressor("low")
        model_low.fit(X_tr, y_tr_low, X_val, y_val_low)
        all_weights["low"].append(model_low.weights.copy())
        
        preds_low = model_low.predict_individual(X_test)
        for key, pred in preds_low.items():
            mae = mean_absolute_error(y_test_low, pred)
            results["low"][key].append(mae)
        
        # === DIRECTION ===
        y_train_dir = targets.loc[train_idx, "direction"]
        y_test_dir = targets.loc[test_idx, "direction"]
        y_tr_dir = y_train_dir.iloc[:val_split]
        y_val_dir = y_train_dir.iloc[val_split:]
        
        model_dir = EnsembleClassifier("direction")
        model_dir.fit(X_tr, y_tr_dir, X_val, y_val_dir)
        all_weights["direction"].append(model_dir.weights.copy())
        
        preds_dir = model_dir.predict_individual(X_test)
        for key, pred in preds_dir.items():
            acc = accuracy_score(y_test_dir, pred)
            results["direction"][key].append(acc)
        
        # Print fold results
        print(f"  High MAE:  XGB={results['high']['xgb'][-1]:.4f}  LGB={results['high']['lgb'][-1]:.4f}  "
              f"CAT={results['high']['cat'][-1]:.4f}  ENS={results['high']['ensemble'][-1]:.4f}")
        print(f"  Low MAE:   XGB={results['low']['xgb'][-1]:.4f}  LGB={results['low']['lgb'][-1]:.4f}  "
              f"CAT={results['low']['cat'][-1]:.4f}  ENS={results['low']['ensemble'][-1]:.4f}")
        print(f"  Dir Acc:   XGB={results['direction']['xgb'][-1]:.2%}  LGB={results['direction']['lgb'][-1]:.2%}  "
              f"CAT={results['direction']['cat'][-1]:.2%}  ENS={results['direction']['ensemble'][-1]:.2%}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (Mean across all folds)")
    print("=" * 70)
    
    summary = {}
    for target in ["high", "low", "direction"]:
        summary[target] = {}
        metric_name = "MAE" if target != "direction" else "Accuracy"
        print(f"\n{target.upper()} ({metric_name}):")
        for model in ["xgb", "lgb", "cat", "ensemble"]:
            mean_val = np.mean(results[target][model])
            std_val = np.std(results[target][model])
            summary[target][model] = {"mean": mean_val, "std": std_val}
            marker = " ⭐" if model == "ensemble" else ""
            print(f"  {model:10}: {mean_val:.4f} ± {std_val:.4f}{marker}")
    
    # Calculate average optimal weights
    avg_weights = {}
    for target in ["high", "low", "direction"]:
        avg_weights[target] = {
            "xgb": np.mean([w["xgb"] for w in all_weights[target]]),
            "lgb": np.mean([w["lgb"] for w in all_weights[target]]),
            "cat": np.mean([w["cat"] for w in all_weights[target]]),
        }
        print(f"\nOptimal weights for {target}:")
        print(f"  XGBoost: {avg_weights[target]['xgb']:.2%}")
        print(f"  LightGBM: {avg_weights[target]['lgb']:.2%}")
        print(f"  CatBoost: {avg_weights[target]['cat']:.2%}")
    
    # Train final models on ALL data
    print("\n" + "=" * 70)
    print("Training Final Ensemble Models on Full Dataset")
    print("=" * 70)
    
    X_all = features[feature_cols]
    
    # Use last 20% for weight optimization
    val_split = int(len(X_all) * 0.8)
    X_tr = X_all.iloc[:val_split]
    X_val = X_all.iloc[val_split:]
    
    # Final High ensemble
    final_high = EnsembleRegressor("high")
    y_tr_high = targets["high_pct"].iloc[:val_split]
    y_val_high = targets["high_pct"].iloc[val_split:]
    final_high.fit(X_tr, y_tr_high, X_val, y_val_high)
    final_high.save(ENSEMBLE_DIR)
    print(f"✅ Saved high ensemble (weights: {final_high.weights})")
    
    # Final Low ensemble
    final_low = EnsembleRegressor("low")
    y_tr_low = targets["low_pct"].iloc[:val_split]
    y_val_low = targets["low_pct"].iloc[val_split:]
    final_low.fit(X_tr, y_tr_low, X_val, y_val_low)
    final_low.save(ENSEMBLE_DIR)
    print(f"✅ Saved low ensemble (weights: {final_low.weights})")
    
    # Final Direction ensemble
    final_dir = EnsembleClassifier("direction")
    y_tr_dir = targets["direction"].iloc[:val_split]
    y_val_dir = targets["direction"].iloc[val_split:]
    final_dir.fit(X_tr, y_tr_dir, X_val, y_val_dir)
    final_dir.save(ENSEMBLE_DIR)
    print(f"✅ Saved direction ensemble (weights: {final_dir.weights})")
    
    # Save metadata
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "n_samples": len(features),
        "n_folds": len(results["high"]["xgb"]),
        "feature_cols": feature_cols,
        "summary": summary,
        "final_weights": {
            "high": final_high.weights,
            "low": final_low.weights,
            "direction": final_dir.weights,
        },
    }
    
    with open(ENSEMBLE_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"\n✅ All models saved to: {ENSEMBLE_DIR}")
    
    # Improvement comparison
    print("\n" + "=" * 70)
    print("IMPROVEMENT vs XGBoost alone")
    print("=" * 70)
    
    for target in ["high", "low"]:
        xgb_mae = np.mean(results[target]["xgb"])
        ens_mae = np.mean(results[target]["ensemble"])
        improvement = (xgb_mae - ens_mae) / xgb_mae * 100
        print(f"{target.upper()}: {improvement:+.2f}% MAE reduction")
    
    xgb_acc = np.mean(results["direction"]["xgb"])
    ens_acc = np.mean(results["direction"]["ensemble"])
    improvement = (ens_acc - xgb_acc) / xgb_acc * 100
    print(f"DIRECTION: {improvement:+.2f}% accuracy improvement")
    
    return summary


if __name__ == "__main__":
    train_ensemble_walk_forward()
