# File: evaluate.py
# Purpose: load the best model, turn scaled sequences into predictions,
#          invert them back to price space, score them, and save two visuals.

import os, json                       # file paths + saving metrics
import numpy as np                    # arrays and vector math
import matplotlib.pyplot as plt       # simple line plot for actual vs predicted
from keras.models import load_model   # load the trained LSTM snapshot
from sklearn.metrics import (         # plain, readable regression metrics
    mean_absolute_error, mean_squared_error, r2_score
)
from dProcess import load_data, create_sequences  # our data helpers
from visualize import plot_candlestick_chart, plot_boxplot  # the two required plots

# ------------- Config -------------
TICKER = "AAPL"               # stock symbol to evaluate
START_DATE = "2010-01-01"     # inclusive start for the dataset
END_DATE   = "2023-12-31"     # exclusive end for the dataset
SEQ_LENGTH = 50               # window size used during training/evaluation
# OHLCV order is ['Open','High','Low','Close','Volume']; Close sits at index 3.
TARGET_INDEX  = 3             # predict 'Close' specifically
GROUPING_DAYS = 3             # each candlestick represents 3 trading days
BOX_WINDOW    = 10            # boxplot uses a rolling 10-day window
OUTPUT_DIR    = "plots"       # where figures and metrics are saved

# ------------- Load data (OHLCV for everything) -------------
# reload_csv=False → use cached CSV if present (faster, avoids re-downloading);
# use_all_features=True → keep the full OHLCV set, not just Close.
scaled_ohlcv_df, ohlcv_df, scaler = load_data(
    ticker=TICKER, start=START_DATE, end=END_DATE,
    reload_csv=False, use_all_features=True
)

# ------------- Build sequences for the model -------------
# create_sequences returns:
#   X_test shape → (N, SEQ_LENGTH, num_features)  e.g., (N, 50, 5)
#   y_test shape → (N,) with the next-step value of our target column
X_test, y_test = create_sequences(scaled_ohlcv_df, SEQ_LENGTH, target_index=TARGET_INDEX)

# ------------- Predict -------------
# load the best checkpoint saved by train.py and produce scaled predictions
model = load_model("models/lstm_model.keras")
pred_scaled = model.predict(X_test)   # shape (N, 1) — still in [0,1] space

# ------------- Inverse scale (place values in the target column) -------------
def inverse_one_column(values_1d, scaler, target_index, n_features):
    """
    Reason: the scaler was fitted on multiple features (OHLCV), but the model
    outputs only one column (our target). MinMaxScaler expects the full width,
    so I pad the missing columns with zeros, invert, then pick the target back.

    Arguments:
      values_1d   : 1-D array of scaled predictions or targets (shape (N,))
      scaler      : the MinMaxScaler fitted on OHLCV
      target_index: which column to place our values into before inversion (3 → Close)
      n_features  : total number of features the scaler knows about (here, 5)

    Returns:
      1-D array of unscaled values corresponding to the chosen target column.
    """
    pad = np.zeros((len(values_1d), n_features), dtype=float)  # make (N, 5) of zeros
    pad[:, target_index] = values_1d                           # drop our values into the correct column
    inv = scaler.inverse_transform(pad)                        # back to price space
    return inv[:, target_index]                                # keep only the target column

# Use the helper for both predictions and ground truth so they share the same path
n_features = scaler.n_features_in_  # sanity: number of columns scaler was fit on
pred_unscaled = inverse_one_column(pred_scaled.reshape(-1), scaler, TARGET_INDEX, n_features)
y_unscaled    = inverse_one_column(y_test.reshape(-1),    scaler, TARGET_INDEX, n_features)

# ------------- Metrics -------------
# Three familiar metrics: absolute error, root mean square error, and R².
mae  = mean_absolute_error(y_unscaled, pred_unscaled)
rmse = np.sqrt(mean_squared_error(y_unscaled, pred_unscaled))
r2   = r2_score(y_unscaled, pred_unscaled)
print(f"MAE : {mae:.4f}\nRMSE: {rmse:.4f}\nR²  : {r2:.4f}")

# Persist the metrics so they can be referenced in the report
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "evaluation_metrics.json"), "w", encoding="utf-8") as f:
    json.dump({"MAE": mae, "RMSE": rmse, "R2": r2}, f, indent=2)

# ------------- Line plot (target = Close) -------------
# A simple overlay of actual vs predicted makes drift or lag obvious at a glance.
plt.figure(figsize=(14,5))
plt.plot(y_unscaled, label='Actual (Close)')
plt.plot(pred_unscaled, label='Predicted (Close)')
plt.title(f'{TICKER} Stock Prediction — Target: Close')
plt.xlabel('Time (Test Sequence Index)')  # index along the test windows
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), dpi=150)
plt.show()

# ------------- Candlestick + Boxplot -------------
# Candlestick: I compress days into blocks of GROUPING_DAYS to make patterns visible over long spans.
#   df            → unscaled OHLCV DataFrame (so prices/volumes are real units)
#   ticker        → label for the title
#   grouping_days → number of trading rows per candle (n ≥ 1)
#   use_trading_days=True → row-based grouping (avoids calendar gaps)
#   mav=(5,10)    → moving averages plotted on top; can be turned off with None
#   figsize       → final size of the figure in inches
#   save_path     → where to write the image
plot_candlestick_chart(
    df=ohlcv_df, ticker=TICKER, grouping_days=GROUPING_DAYS, use_trading_days=True,
    mav=(5,10), figsize=(12,6),
    save_path=os.path.join(OUTPUT_DIR, f"candles_{GROUPING_DAYS}day.png")
)

# Boxplot: each box represents a rolling window of BOX_WINDOW closes, so spread/outliers are obvious.
#   price_column → which series to summarise ('Close' by default)
#   window_size  → length of each rolling window (in trading days)
#   showfliers   → hide outliers to keep the overall shape readable (False here)
plot_boxplot(
    df=ohlcv_df, price_column='Close', window_size=BOX_WINDOW, ticker=TICKER,
    figsize=(15,6), showfliers=False,
    save_path=os.path.join(OUTPUT_DIR, f"boxplot_window{BOX_WINDOW}.png")
)

# A small footer in the console so I know where to look for outputs
print("Saved:", os.path.join(OUTPUT_DIR, "evaluation_metrics.json"))
print("Saved:", os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"))
print("Saved:", os.path.join(OUTPUT_DIR, f"candles_{GROUPING_DAYS}day.png"))
print("Saved:", os.path.join(OUTPUT_DIR, f"boxplot_window{BOX_WINDOW}.png"))
