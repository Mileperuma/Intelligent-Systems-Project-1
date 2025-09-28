# File: evaluate_c5.py
# Purpose: Evaluate a multivariate, multistep model: per-horizon MAE/RMSE, average metrics, and plots.
# Style: I explain decisions as I go, so future me (or a marker) can read intent
#        without opening another document.

import os, json  # file paths + saving metrics
import numpy as np  # arrays and vector math
import matplotlib.pyplot as plt  # simple line plot for actual vs predicted
from keras.models import load_model  # load the trained multistep LSTM snapshot
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dProcess import load_data, split_data, create_multistep_sequences  # our data helpers

# ---------------- Config (match train_c5.py) ----------------
TICKER = "AAPL"  # stock symbol to evaluate
START_DATE = "2010-01-01"  # inclusive start for the dataset
END_DATE = "2023-12-31"  # exclusive end for the dataset
SEQ_LENGTH = 50  # window size used during training/evaluation
HORIZON = 5  # how many future steps to predict (multistep)
TARGET_INDEX = 3  # predict 'Close' specifically in OHLCV
MODEL_PATH = f"models/lstm_ms{HORIZON}.keras"  # model name includes horizon for clarity
OUTPUT_DIR = "plots_c5"  # where figures and metrics are saved

# ---------------- Load & window test set ----------------
# Load OHLCV data once; reload_csv=False → use cached CSV if present (faster)
scaled_ohlcv, raw_ohlcv, scaler = load_data(
    ticker=TICKER, start=START_DATE, end=END_DATE,
    reload_csv=False, use_all_features=True
)

# Split chronologically: first 80% for training, last 20% for testing
# split_by='date' respects time order (no future leakage in evaluation)
train_df, test_df = split_data(scaled_ohlcv, train_ratio=0.8, split_by='date')

# Create multistep sequences: each input window predicts HORIZON future values
# X_test shape → (N, SEQ_LENGTH, 5), Y_test shape → (N, HORIZON)
X_test, Y_test = create_multistep_sequences(
    test_df, SEQ_LENGTH, target_index=TARGET_INDEX, horizon=HORIZON
)

# ---------------- Load model & predict ----------------
model = load_model(MODEL_PATH)
Y_pred = model.predict(X_test)  # shape (N, HORIZON) - predictions for all horizons


# ---------------- Inverse scaling for Close only ----------------
def inverse_target_array(Y_scaled, scaler, target_index=3):
    """
    Reason: the scaler was fitted on multiple features (OHLCV), but our model
    outputs only the target column across multiple horizons. MinMaxScaler expects
    the full feature width, so I flatten, pad missing columns with zeros, invert,
    then reshape back to the original (N, H) shape.

    Arguments:
      Y_scaled     : 2D array of scaled predictions or targets (shape (N, HORIZON))
      scaler       : the MinMaxScaler fitted on OHLCV
      target_index : which column to place our values into before inversion (3 → Close)

    Returns:
      2D array of unscaled values corresponding to the chosen target column.
    """
    n_features = scaler.n_features_in_
    flat = Y_scaled.reshape(-1)  # flatten to (N*H,) for batch processing
    pad = np.zeros((flat.shape[0], n_features), dtype=float)  # create (N*H, 5) zeros
    pad[:, target_index] = flat  # insert our values into the Close column
    inv = scaler.inverse_transform(pad)[:, target_index]  # inverse transform and extract Close
    return inv.reshape(Y_scaled.shape)  # reshape back to (N, HORIZON)


# Apply inverse scaling to both predictions and ground truth using the same method
Y_pred_unscaled = inverse_target_array(Y_pred, scaler, TARGET_INDEX)
Y_true_unscaled = inverse_target_array(Y_test, scaler, TARGET_INDEX)

# ---------------- Metrics (per horizon and averaged) ----------------
# Calculate MAE and RMSE for each prediction horizon separately
# This tells me how prediction quality degrades as we look further into the future
per_step = []
for h in range(HORIZON):
    y_t = Y_true_unscaled[:, h]  # true values for horizon h
    y_p = Y_pred_unscaled[:, h]  # predicted values for horizon h
    mae = mean_absolute_error(y_t, y_p)
    mse = mean_squared_error(y_t, y_p)  # using older sklearn compatible form
    rmse = float(np.sqrt(mse))
    per_step.append({"h": h + 1, "MAE": float(mae), "RMSE": rmse})

# Calculate overall averages across all horizons
avg_mae = float(np.mean([r["MAE"] for r in per_step]))
avg_rmse = float(np.mean([r["RMSE"] for r in per_step]))

# Save metrics to JSON for reporting and later analysis
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, f"metrics_ms{HORIZON}.json"), "w", encoding="utf-8") as f:
    json.dump({"per_horizon": per_step, "avg_MAE": avg_mae, "avg_RMSE": avg_rmse}, f, indent=2)

print("Per-horizon metrics:", per_step)
print(f"Average MAE={avg_mae:.4f}, Average RMSE={avg_rmse:.4f}")


# ---------------- Quick plot (pick a contiguous slice) ----------------
def plot_slice(start=0, length=150):
    """
    Plot a manageable slice of the test data to visualize multistep predictions.
    Shows both ground truth and predictions for each horizon, making it easy to
    see how prediction accuracy changes with increasing horizon.

    Arguments:
      start  : starting index in the test set
      length : how many test examples to plot
    """
    end = min(start + length, Y_true_unscaled.shape[0])  # don't exceed array bounds
    fig = plt.figure(figsize=(12, 5))

    # Plot each horizon separately with distinct styling
    for h in range(HORIZON):
        # Ground truth line
        plt.plot(np.arange(end - start), Y_true_unscaled[start:end, h], label=f"True t+{h + 1}")
        # Predicted dashed line
        plt.plot(np.arange(end - start), Y_pred_unscaled[start:end, h], linestyle="--", label=f"Pred t+{h + 1}")

    plt.title(f"{TICKER} multistep (h={HORIZON}) — window slice [{start}:{end})")
    plt.xlabel("Example index")
    plt.ylabel("Close price")
    plt.legend(ncol=HORIZON, fontsize=8)  # compact legend for multiple horizons
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"multistep_slice_{start}_{end}.png"), dpi=150)
    plt.close(fig)  # free memory after saving


# Generate and save the visualization
plot_slice(0, 150)  # plot first 150 test examples for clarity
print("Saved:", os.path.join(OUTPUT_DIR, f"metrics_ms{HORIZON}.json"))