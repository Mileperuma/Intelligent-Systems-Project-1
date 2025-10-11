# File: ensemble_c6.py
import os, json
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dProcess import load_data, split_data, create_multistep_sequences
from arima_utils import fit_arima, forecast_rolling_matrix

# ---------------- Config (match your C5 defaults) ----------------
TICKER        = "AAPL"
START_DATE    = "2010-01-01"
END_DATE      = "2023-12-31"
SEQ_LENGTH    = 50
HORIZON       = 5
TARGET_INDEX  = 3   # 'Close'
LSTM_MODEL    = f"models/lstm_ms{HORIZON}.keras"
OUT_DIR       = "plots_c6"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Load data (scaled + raw) ----------------
scaled_ohlcv, raw_ohlcv, scaler = load_data(
    ticker=TICKER, start=START_DATE, end=END_DATE,
    reload_csv=False, use_all_features=True
)

# Chronological split (same convention as C5)
train_scaled, test_scaled = split_data(scaled_ohlcv, train_ratio=0.8, split_by='date')

# Align raw with the same indices for Close
train_raw_close = raw_ohlcv.loc[train_scaled.index, "Close"]
test_raw_close  = raw_ohlcv.loc[test_scaled.index,  "Close"]

# ---------------- LSTM predictions (reuse C5 pipeline) ----------------
# Build multistep windows on the *scaled* test slice (same as C5)
X_test, Y_test_scaled = create_multistep_sequences(
    test_scaled, SEQ_LENGTH, target_index=TARGET_INDEX, horizon=HORIZON
)

# Load LSTM and predict scaled → inverse-scale back to Close prices
lstm = load_model(LSTM_MODEL)
Y_pred_lstm_scaled = lstm.predict(X_test)

def inverse_target_matrix(Y_scaled):
    # padding trick to inverse only the target column (OHLCV scaler)
    n_features = scaler.n_features_in_
    flat = Y_scaled.reshape(-1)
    pad = np.zeros((flat.shape[0], n_features))
    pad[:, TARGET_INDEX] = flat
    inv = scaler.inverse_transform(pad)[:, TARGET_INDEX]
    return inv.reshape(Y_scaled.shape)

Y_true = inverse_target_matrix(Y_test_scaled)        # (N, H)
Y_lstm = inverse_target_matrix(Y_pred_lstm_scaled)   # (N, H)

# ---------------- ARIMA predictions on raw Close ----------------
# Fit ARIMA on *training* close
arima_res = fit_arima(train_raw_close, order=(5,1,0))

# Forecast over the entire test span + horizon, then slice into (N, H)
# N must match the LSTM's N from create_multistep_sequences
test_len = len(test_raw_close)
N = Y_true.shape[0]  # number of LSTM windows
# Our helper builds (test_len - H + 1, H). We align the first N rows.
arima_mat_full = forecast_rolling_matrix(arima_res, test_len=test_len, horizon=HORIZON)
Y_arima = arima_mat_full[-N:] if arima_mat_full.shape[0] >= N else arima_mat_full[:N]

# ---------------- Ensemble: simple average (weights can be tuned later) ----------------
def ensemble_avg(A, B, wA=0.5):
    wB = 1.0 - wA
    return wA * A + wB * B

Y_ens = ensemble_avg(Y_lstm, Y_arima, wA=0.5)

# ---------------- Metrics (per horizon + averages) ----------------
def per_horizon_scores(Y_true, Y_hat):
    rows = []
    for h in range(Y_true.shape[1]):
        y_t, y_p = Y_true[:, h], Y_hat[:, h]
        mae = float(mean_absolute_error(y_t, y_p))
        rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
        rows.append({"h": h+1, "MAE": mae, "RMSE": rmse})
    avg_mae  = float(np.mean([r["MAE"] for r in rows]))
    avg_rmse = float(np.mean([r["RMSE"] for r in rows]))
    return rows, avg_mae, avg_rmse

scores_lstm, lstm_mae, lstm_rmse = per_horizon_scores(Y_true, Y_lstm)
scores_arima, arima_mae, arima_rmse = per_horizon_scores(Y_true, Y_arima)
scores_ens, ens_mae, ens_rmse = per_horizon_scores(Y_true, Y_ens)

metrics = {
    "LSTM": {"per_horizon": scores_lstm, "avg_MAE": lstm_mae, "avg_RMSE": lstm_rmse},
    "ARIMA": {"per_horizon": scores_arima, "avg_MAE": arima_mae, "avg_RMSE": arima_rmse},
    "Ensemble(avg)": {"per_horizon": scores_ens, "avg_MAE": ens_mae, "avg_RMSE": ens_rmse},
}
with open(os.path.join(OUT_DIR, f"c6_metrics_ms{HORIZON}.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))

# ---------------- Visual: compare lines for each horizon (one slice) ----------------
def plot_slice(start=0, length=150):
    end = min(start + length, Y_true.shape[0])
    fig = plt.figure(figsize=(12, 6))
    for h in range(HORIZON):
        x = np.arange(end - start)
        plt.plot(x, Y_true[start:end, h], label=f"True t+{h+1}")
        plt.plot(x, Y_lstm[start:end, h], linestyle="--", label=f"LSTM t+{h+1}")
        plt.plot(x, Y_arima[start:end, h], linestyle=":",  label=f"ARIMA t+{h+1}")
        plt.plot(x, Y_ens[start:end, h], linestyle="-.",   label=f"Ens t+{h+1}")
    plt.title(f"{TICKER} Ensemble (H={HORIZON}) — slice [{start}:{end})")
    plt.xlabel("Example index")
    plt.ylabel("Close price")
    plt.legend(ncol=4, fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"ensemble_slice_{start}_{end}.png"), dpi=150)
    plt.close(fig)

plot_slice(0, 150)
print("Saved:",
      os.path.join(OUT_DIR, f"c6_metrics_ms{HORIZON}.json"),
      os.path.join(OUT_DIR, f"ensemble_slice_0_150.png"))
