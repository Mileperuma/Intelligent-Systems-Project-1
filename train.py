# File: train.py
# Purpose: train an LSTM on OHLCV data with Close as the target, and keep the best snapshot.

import os                              # paths and folders for saving model/losses
import numpy as np                     # saving loss curves as .npy files
from keras.callbacks import ModelCheckpoint   # checkpoint the best model during training
from dProcess import load_data, split_data, create_sequences  # my data helpers (download, split, window)
from model import build_lstm_model     # the network factory (two LSTMs + Dropout + Dense)

# ---------------- Configuration ----------------
TICKER        = "AAPL"                 # which instrument to train on
START_DATE    = "2022-01-01"           # inclusive start for the training dataset
END_DATE      = "2023-12-31"           # exclusive end for the training dataset
SEQ_LENGTH    = 50                     # number of past timesteps per training example
EPOCHS        = 25                     # training passes over the dataset
BATCH_SIZE    = 32                     # mini-batch size for gradient updates
TRAIN_RATIO   = 0.8                    # 80% chronological train, 20% validation
MODEL_DIR     = "models"               # where to save the checkpoint and curves
MODEL_NAME    = "lstm_model.keras"     # filename for Keras SavedModel

# OHLCV column order used throughout:
# ['Open','High','Low','Close','Volume'] → Close index is 3
TARGET_INDEX  = 3                      # we predict the 'Close' column specifically

# ---------------- Load + split ----------------
# load_data:
#   - ticker/start/end: define the time span to fetch from Yahoo Finance (or load from cache)
#   - reload_csv=True: force a fresh download/csv so the span matches the config exactly
#   - use_all_features=True: keep all five OHLCV columns (not just Close)
# returns (scaled_df, raw_df, scaler) → I train on the scaled_df.
scaled_ohlcv, raw_ohlcv, scaler = load_data(
    ticker=TICKER, start=START_DATE, end=END_DATE,
    reload_csv=True,                 # ensure fresh CSV with all columns
    use_all_features=True
)

# split_data:
#   - train_ratio=0.8: first 80% of rows → train, remaining 20% → validation (chronological)
#   - split_by='date': respect time order (no leakage from the future)
train_df, val_df = split_data(scaled_ohlcv, train_ratio=TRAIN_RATIO, split_by='date')

# ---------------- Windowing ----------------
# create_sequences:
#   - df: scaled features (shape: T×5)
#   - SEQ_LENGTH: window length (e.g., 50 timesteps per example)
#   - target_index=3: the index of 'Close' within OHLCV
# returns:
#   X_* → (N, SEQ_LENGTH, 5), y_* → (N,)
X_train, y_train = create_sequences(train_df, SEQ_LENGTH, target_index=TARGET_INDEX)
X_val,   y_val   = create_sequences(val_df,   SEQ_LENGTH, target_index=TARGET_INDEX)

# ---------------- Build model ----------------
# input_shape must be (timesteps, features); taken directly from X_train
input_shape = (X_train.shape[1], X_train.shape[2])   # e.g., (50, 5)
model = build_lstm_model(input_shape)                 # two LSTMs + Dropout + Dense(1), 'adam' + 'mse'

# ---------------- Train (+ checkpoint best on val loss) ----------------
os.makedirs(MODEL_DIR, exist_ok=True)                 # create the save folder once; no error if exists

# ModelCheckpoint:
#   - filepath: where to store the best-so-far snapshot
#   - save_best_only=True: only overwrite when validation improves
#   - monitor='val_loss': optimisation target during training is MSE on the validation split
#   - verbose=1: print when a new best model is saved
ckpt = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, MODEL_NAME),
    save_best_only=True,
    monitor='val_loss',
    verbose=1
)

# model.fit:
#   - X_train, y_train: supervised pairs
#   - epochs/batch_size: training schedule
#   - validation_data: evaluate after each epoch on held-out tail
#   - callbacks=[ckpt]: keep the best weights according to val_loss
#   - verbose=1: progress + per-epoch metrics in the console
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[ckpt],
    verbose=1
)

# ---------------- Save loss curves ----------------
# Stash the learning curves so I can plot them later or include them in the report.
np.save(os.path.join(MODEL_DIR, "train_loss.npy"), history.history["loss"])
np.save(os.path.join(MODEL_DIR, "val_loss.npy"),   history.history["val_loss"])

print("Training complete. Best model saved to:", os.path.join(MODEL_DIR, MODEL_NAME))
