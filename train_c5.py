# File: train_c5.py
# Purpose: Train a multivariate, multistep model (C5) without breaking C4 scripts.
# Style: I explain decisions as I go, so future me (or a marker) can read intent
#        without opening another document.

import os
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from dProcess import load_data, split_data, create_multistep_sequences
from model import build_dl_model

# ---------------- Configuration (tweak as needed) ----------------
TICKER        = "AAPL"               # stock symbol to train on
START_DATE    = "2010-01-01"         # inclusive start for the dataset
END_DATE      = "2023-12-31"         # exclusive end for the dataset
SEQ_LENGTH    = 50                   # window length (past timesteps per example)
HORIZON       = 5                    # number of future steps to predict (multistep)
EPOCHS        = 25                   # training passes over the dataset
BATCH_SIZE    = 32                   # mini-batch size for gradient updates
TRAIN_RATIO   = 0.8                  # 80% chronological train, 20% validation
MODEL_DIR     = "models"             # where to save the checkpoint and curves
MODEL_NAME    = f"lstm_ms{HORIZON}.keras"   # filename includes horizon for clarity
LAYER_TYPE    = "LSTM"               # recurrent layer type
NUM_LAYERS    = 2                    # number of stacked recurrent layers
UNITS         = 64                   # hidden units per layer
DROPOUT       = 0.2                  # dropout rate for regularization
BIDIRECTIONAL = False                # whether to use bidirectional layers

TARGET_INDEX  = 3                    # index of 'Close' in OHLCV ['Open','High','Low','Close','Volume']

# ---------------- Load + split ----------------
# Load OHLCV data; reload_csv=False → use cached CSV if present (faster)
scaled_ohlcv, raw_ohlcv, scaler = load_data(
    ticker=TICKER, start=START_DATE, end=END_DATE,
    reload_csv=False, use_all_features=True
)

# Split chronologically: first 80% for training, last 20% for validation
# split_by='date' respects time order (no future leakage in validation)
train_df, val_df = split_data(scaled_ohlcv, train_ratio=TRAIN_RATIO, split_by='date')

# ---------------- Windowing (multistep) ----------------
# Create sequences where each input window predicts HORIZON future values
# X_train shape → (N, SEQ_LENGTH, 5), Y_train shape → (N, HORIZON)
X_train, Y_train = create_multistep_sequences(train_df, SEQ_LENGTH, target_index=TARGET_INDEX, horizon=HORIZON)
X_val,   Y_val   = create_multistep_sequences(val_df,   SEQ_LENGTH, target_index=TARGET_INDEX, horizon=HORIZON)

input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features), e.g., (50, 5)

# ---------------- Build model (k outputs) ----------------
# Key change from single-step: dense_units=HORIZON outputs multiple future predictions
model = build_dl_model(
    input_shape=input_shape,
    layer_type=LAYER_TYPE,
    num_layers=NUM_LAYERS,
    units=UNITS,
    dropout=DROPOUT,
    bidirectional=BIDIRECTIONAL,
    dense_units=HORIZON,   # crucial: multi-output head for multistep prediction
    loss="mse",            # mean squared error works for multi-output regression
    optimizer="adam"       # robust default optimizer
)

# ---------------- Train (+ early stop + best checkpoint) ----------------
os.makedirs(MODEL_DIR, exist_ok=True)  # create save folder if needed

# ModelCheckpoint saves only the best weights (lowest validation loss)
ckpt = ModelCheckpoint(os.path.join(MODEL_DIR, MODEL_NAME), save_best_only=True, monitor='val_loss', verbose=1)
# EarlyStopping prevents overfitting by stopping when validation stops improving
es   = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Train the model with validation monitoring
history = model.fit(
    X_train, Y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, Y_val),  # evaluate on held-out validation set
    callbacks=[ckpt, es],            # save best model and stop early if needed
    verbose=1                        # show progress bars and metrics
)

# Save loss curves for later analysis and reporting
np.save(os.path.join(MODEL_DIR, f"train_loss_ms{HORIZON}.npy"), history.history["loss"])
np.save(os.path.join(MODEL_DIR, f"val_loss_ms{HORIZON}.npy"),   history.history["val_loss"])

print(f"Training complete. Best multistep model saved to: {os.path.join(MODEL_DIR, MODEL_NAME)}")