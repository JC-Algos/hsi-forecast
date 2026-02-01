#!/usr/bin/env python3
"""
Visualization and analysis for HSI forecasting model.
Generates charts for model evaluation and optimization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

PROJECT_DIR = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"
CHARTS_DIR = PROJECT_DIR / "charts"

# Style settings
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def load_data():
    """Load processed data and predictions."""
    features = pd.read_csv(PROCESSED_DIR / "features.csv", index_col=0, parse_dates=True)
    targets = pd.read_csv(PROCESSED_DIR / "targets.csv", index_col=0, parse_dates=True)
    predictions = pd.read_csv(PROCESSED_DIR / "backtest_predictions.csv", index_col=0, parse_dates=True)
    return features, targets, predictions


def plot_fitting_charts(predictions: pd.DataFrame, save_path: Path):
    """Plot actual vs predicted for all targets."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. High % - Actual vs Predicted
    ax = axes[0, 0]
    ax.scatter(predictions['actual_high_pct'], predictions['pred_high_pct'], alpha=0.5, s=20)
    max_val = max(predictions['actual_high_pct'].max(), predictions['pred_high_pct'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Fit')
    ax.set_xlabel('Actual High %')
    ax.set_ylabel('Predicted High %')
    ax.set_title('Range High: Actual vs Predicted')
    ax.legend()
    
    # 2. Low % - Actual vs Predicted
    ax = axes[0, 1]
    ax.scatter(predictions['actual_low_pct'], predictions['pred_low_pct'], alpha=0.5, s=20)
    min_val = min(predictions['actual_low_pct'].min(), predictions['pred_low_pct'].min())
    ax.plot([min_val, 0], [min_val, 0], 'r--', lw=2, label='Perfect Fit')
    ax.set_xlabel('Actual Low %')
    ax.set_ylabel('Predicted Low %')
    ax.set_title('Range Low: Actual vs Predicted')
    ax.legend()
    
    # 3. High % - Time Series
    ax = axes[1, 0]
    ax.plot(predictions.index[-100:], predictions['actual_high_pct'].iloc[-100:], 
            label='Actual', alpha=0.7, linewidth=1.5)
    ax.plot(predictions.index[-100:], predictions['pred_high_pct'].iloc[-100:], 
            label='Predicted', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('High %')
    ax.set_title('Range High: Last 100 Days')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    
    # 4. Low % - Time Series
    ax = axes[1, 1]
    ax.plot(predictions.index[-100:], predictions['actual_low_pct'].iloc[-100:], 
            label='Actual', alpha=0.7, linewidth=1.5)
    ax.plot(predictions.index[-100:], predictions['pred_low_pct'].iloc[-100:], 
            label='Predicted', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Low %')
    ax.set_title('Range Low: Last 100 Days')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path / 'fitting_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path / 'fitting_chart.png'}")


def plot_direction_analysis(predictions: pd.DataFrame, save_path: Path):
    """Plot direction prediction analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Confusion Matrix Heatmap
    ax = axes[0, 0]
    cm = pd.crosstab(predictions['actual_direction'], predictions['pred_direction'], 
                     rownames=['Actual'], colnames=['Predicted'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'])
    ax.set_title('Direction Confusion Matrix')
    
    # 2. Probability Distribution by Actual Direction
    ax = axes[0, 1]
    down_probs = predictions[predictions['actual_direction'] == 0]['pred_direction_prob']
    up_probs = predictions[predictions['actual_direction'] == 1]['pred_direction_prob']
    ax.hist(down_probs, bins=30, alpha=0.6, label='Actual DOWN', color='red')
    ax.hist(up_probs, bins=30, alpha=0.6, label='Actual UP', color='green')
    ax.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    ax.set_xlabel('Predicted Probability (UP)')
    ax.set_ylabel('Frequency')
    ax.set_title('Probability Distribution by Actual Direction')
    ax.legend()
    
    # 3. Accuracy by Confidence Level
    ax = axes[1, 0]
    predictions['confidence'] = np.abs(predictions['pred_direction_prob'] - 0.5) * 2
    predictions['correct'] = (predictions['actual_direction'] == predictions['pred_direction']).astype(int)
    
    conf_bins = np.arange(0, 1.1, 0.1)
    conf_accuracy = []
    conf_counts = []
    for i in range(len(conf_bins) - 1):
        mask = (predictions['confidence'] >= conf_bins[i]) & (predictions['confidence'] < conf_bins[i+1])
        if mask.sum() > 0:
            conf_accuracy.append(predictions.loc[mask, 'correct'].mean())
            conf_counts.append(mask.sum())
        else:
            conf_accuracy.append(0)
            conf_counts.append(0)
    
    x_pos = np.arange(len(conf_bins) - 1)
    bars = ax.bar(x_pos, conf_accuracy, color='steelblue', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{conf_bins[i]:.1f}-{conf_bins[i+1]:.1f}' for i in range(len(conf_bins)-1)], rotation=45)
    ax.set_xlabel('Confidence Level')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Confidence Level')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.set_ylim(0, 1.1)
    
    # Add count labels
    for bar, count in zip(bars, conf_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'n={count}', ha='center', va='bottom', fontsize=8)
    
    # 4. Cumulative Accuracy Over Time
    ax = axes[1, 1]
    predictions['cum_correct'] = predictions['correct'].cumsum()
    predictions['cum_count'] = np.arange(1, len(predictions) + 1)
    predictions['cum_accuracy'] = predictions['cum_correct'] / predictions['cum_count']
    
    ax.plot(predictions.index, predictions['cum_accuracy'], linewidth=1.5, color='steelblue')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Accuracy')
    ax.set_title('Cumulative Direction Accuracy Over Time')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path / 'direction_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path / 'direction_analysis.png'}")


def plot_feature_correlation(features: pd.DataFrame, targets: pd.DataFrame, save_path: Path):
    """Plot feature correlation heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Feature-Feature Correlation
    ax = axes[0]
    corr = features.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title('Feature Correlation Matrix')
    
    # 2. Feature-Target Correlation
    ax = axes[1]
    combined = features.join(targets[['high_pct', 'low_pct', 'direction']])
    target_corr = combined.corr()[['high_pct', 'low_pct', 'direction']].drop(['high_pct', 'low_pct', 'direction'])
    sns.heatmap(target_corr, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                center=0, ax=ax, linewidths=0.5)
    ax.set_title('Feature-Target Correlation')
    
    plt.tight_layout()
    plt.savefig(save_path / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path / 'correlation_heatmap.png'}")


def plot_feature_importance(save_path: Path):
    """Plot feature importance comparison."""
    importance = pd.read_csv(MODELS_DIR / "feature_importance.csv")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(importance))
    width = 0.25
    
    bars1 = ax.barh(x - width, importance['importance_high'], width, label='High Model', alpha=0.8)
    bars2 = ax.barh(x, importance['importance_low'], width, label='Low Model', alpha=0.8)
    bars3 = ax.barh(x + width, importance['importance_dir'], width, label='Direction Model', alpha=0.8)
    
    ax.set_yticks(x)
    ax.set_yticklabels(importance['feature'])
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance Comparison')
    ax.legend()
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path / 'feature_importance.png'}")


def run_hyperparameter_optimization(features: pd.DataFrame, targets: pd.DataFrame, save_path: Path):
    """Run grid search for hyperparameter optimization."""
    print("\n" + "=" * 60)
    print("Hyperparameter Optimization (High Range Model)")
    print("=" * 60)
    
    with open(MODELS_DIR / "metadata.json") as f:
        metadata = json.load(f)
    feature_cols = metadata["feature_cols"]
    
    X = features[feature_cols]
    y = targets["high_pct"]
    
    # Parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
    }
    
    model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("Running GridSearchCV...")
    grid_search = GridSearchCV(
        model, param_grid, cv=tscv, scoring='neg_mean_absolute_error',
        n_jobs=-1, verbose=1
    )
    grid_search.fit(X, y)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best MAE: {-grid_search.best_score_:.4f}%")
    
    # Create heatmap of results
    results = pd.DataFrame(grid_search.cv_results_)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, lr in enumerate([0.05, 0.1, 0.2]):
        ax = axes[idx]
        subset = results[results['param_learning_rate'] == lr]
        pivot = subset.pivot_table(
            values='mean_test_score',
            index='param_n_estimators',
            columns='param_max_depth'
        )
        pivot = -pivot  # Convert to positive MAE
        
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax)
        ax.set_title(f'MAE Heatmap (learning_rate={lr})')
        ax.set_xlabel('max_depth')
        ax.set_ylabel('n_estimators')
    
    plt.tight_layout()
    plt.savefig(save_path / 'optimization_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path / 'optimization_heatmap.png'}")
    
    # Save best params
    with open(MODELS_DIR / "best_params.json", "w") as f:
        json.dump(grid_search.best_params_, f, indent=2)
    print(f"Saved: {MODELS_DIR / 'best_params.json'}")
    
    return grid_search.best_params_


def plot_residual_analysis(predictions: pd.DataFrame, save_path: Path):
    """Plot residual analysis for range models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Calculate residuals
    predictions['high_residual'] = predictions['actual_high_pct'] - predictions['pred_high_pct']
    predictions['low_residual'] = predictions['actual_low_pct'] - predictions['pred_low_pct']
    
    # 1. High Residual Distribution
    ax = axes[0, 0]
    ax.hist(predictions['high_residual'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual (Actual - Predicted)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'High % Residual Distribution (μ={predictions["high_residual"].mean():.3f})')
    
    # 2. Low Residual Distribution
    ax = axes[0, 1]
    ax.hist(predictions['low_residual'], bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual (Actual - Predicted)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Low % Residual Distribution (μ={predictions["low_residual"].mean():.3f})')
    
    # 3. High Residual vs Predicted
    ax = axes[1, 0]
    ax.scatter(predictions['pred_high_pct'], predictions['high_residual'], alpha=0.5, s=20)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted High %')
    ax.set_ylabel('Residual')
    ax.set_title('High % Residual vs Predicted (Homoscedasticity Check)')
    
    # 4. Low Residual vs Predicted
    ax = axes[1, 1]
    ax.scatter(predictions['pred_low_pct'], predictions['low_residual'], alpha=0.5, s=20)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted Low %')
    ax.set_ylabel('Residual')
    ax.set_title('Low % Residual vs Predicted (Homoscedasticity Check)')
    
    plt.tight_layout()
    plt.savefig(save_path / 'residual_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path / 'residual_analysis.png'}")


def main():
    """Generate all visualizations."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("HSI Forecast Model Visualization")
    print("=" * 60)
    
    # Load data
    features, targets, predictions = load_data()
    print(f"Loaded {len(features)} samples")
    
    # Generate all plots
    print("\nGenerating charts...")
    plot_fitting_charts(predictions, CHARTS_DIR)
    plot_direction_analysis(predictions, CHARTS_DIR)
    plot_feature_correlation(features, targets, CHARTS_DIR)
    plot_feature_importance(CHARTS_DIR)
    plot_residual_analysis(predictions, CHARTS_DIR)
    
    # Run optimization
    best_params = run_hyperparameter_optimization(features, targets, CHARTS_DIR)
    
    print("\n" + "=" * 60)
    print(f"All charts saved to: {CHARTS_DIR}")
    print("=" * 60)
    
    return best_params


if __name__ == "__main__":
    main()
