# Stock Price Prediction with LSTM & ARIMA Ensemble

A multi-stage deep learning project for forecasting AAPL stock prices using recurrent neural networks and a classical ARIMA baseline. Built in Python with TensorFlow/Keras, this project progresses from a single-step LSTM baseline through multistep forecasting, architecture grid search, and a hybrid LSTM–ARIMA ensemble.

---

## What It Does

The project is structured as a series of experiments, each building on the last:

| Stage | Script(s) | Description |
|-------|-----------|-------------|
| Baseline LSTM | `train.py` / `evaluate.py` | Single-step next-day Close price prediction with a stacked LSTM on OHLCV data |
| Architecture Search | `train_grid.py` | Grid search over LSTM/GRU/RNN variants; results logged to `experiments/summary.csv` |
| Multistep Forecasting | `train_c5.py` / `evaluate_c5.py` | Predict the next *k* days simultaneously (multi-output LSTM head) |
| LSTM–ARIMA Ensemble | `ensemble_c6.py` | Average the LSTM multistep forecasts with a rolling ARIMA forecast; compare per-horizon MAE/RMSE |

**Visualisations produced:**
- Actual vs Predicted line chart
- 3-day grouped candlestick chart with volume and moving averages
- Rolling 10-day boxplot of Close prices
- Per-horizon prediction overlays (multistep & ensemble)

---

## Project Structure

```
.
├── dProcess.py          # Data pipeline: download, cache, scale, split, window
├── model.py             # Model factory: LSTM / GRU / RNN, configurable depth & width
├── visualize.py         # Candlestick chart and rolling boxplot helpers
├── arima_utils.py       # ARIMA fit and rolling forecast matrix
│
├── train.py             # Train single-step LSTM baseline
├── evaluate.py          # Evaluate baseline; produces plots/ outputs
│
├── train_grid.py        # Architecture grid search (C4)
│
├── train_c5.py          # Train multistep LSTM (C5)
├── evaluate_c5.py       # Evaluate multistep model; produces plots_c5/ outputs
│
├── ensemble_c6.py       # LSTM + ARIMA ensemble evaluation (C6)
│
├── stock_prediction.py  # Provided starter/template code (tutorial baseline)
│
├── datasets/            # Cached CSV downloads and fitted scaler (.pkl)
├── models/              # Saved model checkpoints (.keras)
├── plots/               # Output figures and metrics JSON
│
└── P1/                  # Original tutorial code provided as the project starting point
```

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| TensorFlow / Keras | LSTM, GRU, SimpleRNN model building and training |
| scikit-learn | MinMaxScaler, train/test split, MAE / RMSE / R² metrics |
| pandas / NumPy | Data wrangling and sliding-window construction |
| yfinance | Yahoo Finance OHLCV data download |
| statsmodels | ARIMA model fitting and forecasting |
| matplotlib | Line plots, candlestick charts, boxplots |
| mplfinance | OHLCV candlestick rendering |

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

Each script is self-contained. Run them in order to reproduce the full experiment sequence.

### 1. Train the baseline LSTM

```bash
python train.py
```

Downloads AAPL data (2022–2023), trains a 2-layer LSTM on OHLCV features, saves the best checkpoint to `models/lstm_model.keras`.

### 2. Evaluate the baseline

```bash
python evaluate.py
```

Loads the saved model, produces an actual-vs-predicted plot, a candlestick chart, a rolling boxplot, and saves metrics to `plots/evaluation_metrics.json`.

### 3. Run the architecture grid search

```bash
python train_grid.py
```

Trains 5 RNN variants (LSTM, GRU, SimpleRNN, deeper LSTM, bidirectional GRU). Saves results to `experiments/summary.csv` for comparison.

### 4. Train the multistep model

```bash
python train_c5.py
```

Trains a 5-step-ahead LSTM (horizon = 5 days). Saves checkpoint to `models/lstm_ms5.keras`.

### 5. Evaluate multistep model

```bash
python evaluate_c5.py
```

Reports per-horizon MAE/RMSE (h=1 through h=5) and saves a prediction overlay plot to `plots_c5/`.

### 6. Run the LSTM–ARIMA ensemble

```bash
python ensemble_c6.py
```

Fits ARIMA(5,1,0) on the training Close series, averages its rolling forecasts with the LSTM predictions, and compares all three (LSTM / ARIMA / Ensemble) across each horizon. Saves metrics and plots to `plots_c6/`.

---

## Configuration

All hyperparameters are defined as named constants at the top of each script — no config file or CLI flags needed. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TICKER` | `"AAPL"` | Stock symbol |
| `START_DATE` / `END_DATE` | 2010-01-01 / 2023-12-31 | Training data window |
| `SEQ_LENGTH` | 50 | Look-back window (timesteps per input) |
| `HORIZON` | 5 | Forecast horizon for multistep models |
| `TRAIN_RATIO` | 0.8 | Chronological train/validation split |

---

## Results

Baseline single-step LSTM on AAPL (2010–2023, 20% test split):

| Metric | Value |
|--------|-------|
| MAE | see `plots/evaluation_metrics.json` |
| RMSE | see `plots/evaluation_metrics.json` |
| R² | see `plots/evaluation_metrics.json` |

Ensemble vs LSTM (multistep, horizon 1–5): see `plots_c6/c6_metrics_ms5.json` after running `ensemble_c6.py`.

---

## Notes

- Data is downloaded from Yahoo Finance via `yfinance` and cached locally. Set `reload_csv=True` in `load_data()` to force a fresh download.
- The scaler is fitted only on training data to prevent leakage into the test set.
- The `P1/` directory contains the original tutorial code that was provided as the project starting point and is kept for reference.
