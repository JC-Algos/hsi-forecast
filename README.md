# HSI Daily Range & Direction Forecast

XGBoost-based model to predict daily Hang Seng Index trading range and direction probability before market open.

## Performance (Backtest)

| Metric | Value |
|--------|-------|
| High Range MAE | 0.50% |
| Low Range MAE | 0.48% |
| Direction Accuracy | 72.3% |
| Direction AUC-ROC | 0.81 |

## Features (16 total)

| Feature | Importance | Description |
|---------|------------|-------------|
| **fxi_change_pct** | 10.3% | FXI ETF overnight change (best HK proxy!) |
| day_of_week | 9.0% | Monday effect, etc. |
| hsi_volatility | 8.3% | 10-day rolling volatility |
| spx_close | 8.3% | S&P 500 close |
| usdcnh | 6.6% | USD/CNH exchange rate |
| hsi_prev_close | 6.6% | Previous HSI close |
| ndx_change_pct | 6.5% | Nasdaq overnight change |
| hsi_ma5 | 6.4% | 5-day moving average |
| ndx_close | 5.8% | Nasdaq close |
| fxi_close | 5.7% | FXI ETF close level |
| vix | 5.3% | CBOE Volatility Index |
| usdcnh_change_pct | 5.3% | Currency change |
| vix_change_pct | 5.2% | VIX change (fear spike) |
| hsi_ema20 | 4.2% | 20-day EMA |
| spx_change_pct | 3.4% | S&P overnight change |
| hsi_change_pct | 3.0% | Previous day HSI change |

## Quick Start

```bash
# 1. Fetch 3 years of historical data
python3 src/fetch_data.py

# 2. Create features and targets
python3 src/features.py

# 3. Train models with walk-forward validation
python3 src/train.py

# 4. Run backtest analysis
python3 src/backtest.py

# 5. Get today's prediction
python3 src/predict.py
```

## Prediction Output

```json
{
  "date": "2026-02-03",
  "reference_price": 27387.11,
  "predicted_high": 27650,
  "predicted_high_pct": 0.96,
  "predicted_low": 27120,
  "predicted_low_pct": -0.98,
  "predicted_range": 530,
  "direction": "DOWN",
  "direction_prob": 0.35,
  "volatility_regime": "NORMAL",
  "gap_adjustment": "FXI -1.2% → gap down"
}
```

## Model Architecture

```
┌─────────────────────────────────────────────────┐
│                 16 Features                      │
└─────────────────┬───────────────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌────────┐  ┌────────┐  ┌──────────────┐
│ XGBoost│  │ XGBoost│  │   XGBoost    │
│  Regr  │  │  Regr  │  │  Classifier  │
└───┬────┘  └───┬────┘  └──────┬───────┘
    ▼           ▼              ▼
  High %      Low %      Direction Prob
 from close  from close   (0.0 - 1.0)
```

## Gap Adjustment

The model includes a post-prediction gap adjustment based on:
- **FXI overnight move**: Primary indicator (FXI trades in US hours)
- **VIX level & change**: Fear gauge for volatility scaling

## Validation

Walk-forward with expanding window:
- Initial training: 18 months
- Test window: 1 month
- ~12 independent test periods over 3 years

## Data Sources

| Ticker | Description |
|--------|-------------|
| ^HSI | Hang Seng Index |
| ^GSPC | S&P 500 |
| ^IXIC | Nasdaq Composite |
| CNH=F | USD/CNH Futures |
| ^VIX | CBOE Volatility Index |
| FXI | iShares China Large Cap ETF |

## Project Structure

```
hsi-forecast/
├── data/
│   ├── raw/           # Downloaded OHLCV data
│   └── processed/     # Features & targets
├── models/
│   ├── range_high.json    # XGBoost High model
│   ├── range_low.json     # XGBoost Low model
│   ├── direction.json     # XGBoost Direction model
│   └── metadata.json      # Feature columns, params
├── charts/            # Visualization outputs
├── src/
│   ├── fetch_data.py  # Data download
│   ├── features.py    # Feature engineering
│   ├── train.py       # Walk-forward training
│   ├── predict.py     # Daily prediction
│   ├── backtest.py    # Performance analysis
│   └── visualize.py   # Charts & optimization
└── README.md
```

## Requirements

```
yfinance
pandas
numpy
xgboost
scikit-learn
matplotlib
seaborn
```

## License

MIT
