# File: train_grid.py
# Purpose: small experiment runner for Task C.4 — build several RNN variants from a
#          compact config (layer type, depth, units, dropout, etc.), train each one,
#          and log results to a single CSV so I can compare fairly.

import os
import json
import time
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from dProcess import load_data, split_data, create_sequences
from model import build_dl_model  # the model factory added for C.4


# --------- Data config (reuse v0.3 settings) ----------
TICKER      = "AAPL"          # instrument to evaluate across runs
START_DATE  = "2010-01-01"    # inclusive start of dataset
END_DATE    = "2023-12-31"    # exclusive end of dataset
SEQ_LENGTH  = 50              # window length used to build (X, y)
TARGET_IDX  = 3               # index of 'Close' in OHLCV → ['Open','High','Low','Close','Volume']


# --------- Experiment output locations ---------
MODEL_DIR   = "models"        # every trained run saves a .keras snapshot here
EXP_DIR     = "experiments"   # where I keep the table that summarises runs
EXP_CSV     = os.path.join(EXP_DIR, "summary.csv")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)


# --------- Load + prepare once (shared across all runs) ---------
# I load and window the data a single time; each model then trains on the exact same arrays,
# which makes the comparison fair (no accidental differences between runs).
scaled, raw, scaler = load_data(
    TICKER, START_DATE, END_DATE,
    reload_csv=False,            # reuse cached CSV if available; faster and consistent
    use_all_features=True        # keep OHLCV because the visualisations also expect them
)
train_df, val_df   = split_data(scaled, train_ratio=0.8, split_by='date')
X_train, y_train   = create_sequences(train_df, SEQ_LENGTH, target_index=TARGET_IDX)
X_val,   y_val     = create_sequences(val_df,   SEQ_LENGTH, target_index=TARGET_IDX)
input_shape        = (X_train.shape[1], X_train.shape[2])  # (timesteps, features), e.g., (50, 5)


# --------- Define a small grid (tweak as needed) ---------
# Each tuple fully describes one experiment:
#   (layer_type, num_layers, units, dropout, bidir, epochs, batch)
# - layer_type: "LSTM" | "GRU" | "RNN"
# - num_layers: how many recurrent layers to stack
# - units: int or list[int]. If list, it's per-layer widths (e.g., [64,64,32])
# - dropout: Dropout after each recurrent layer
# - bidir: whether to wrap layers with Bidirectional
# - epochs / batch: training schedule
grid = [
    ("LSTM", 2, 64,        0.2, False, 15, 32),
    ("GRU",  2, 64,        0.2, False, 15, 32),
    ("RNN",  2, 64,        0.2, False, 15, 32),
    ("LSTM", 3, [64,64,32],0.2, False, 20, 32),
    ("GRU",  2, 128,       0.2, True,  20, 64),
]


def run_one(cfg):
    """
    Train a single configuration and return a dict with metrics and provenance.

    Parameters:
      cfg: tuple matching the schema above:
           (layer_type, num_layers, units, dropout, bidir, epochs, batch)

    Returns:
      dict with:
        - tag: readable experiment name (used in filenames)
        - layer_type, num_layers, units, dropout, bidirectional
        - epochs_ran (may be < requested when EarlyStopping fires)
        - batch_size
        - best_val_loss (the one we compare across runs)
        - final_train_loss (for sanity vs. val — detects overfitting)
        - model_path (where the best snapshot was saved)
        - timestamp (so I can sort runs later)
    """
    layer_type, num_layers, units, dropout, bidir, epochs, batch = cfg

    # Create a compact human-readable tag for this run, e.g.:
    # "GRU_L2_U64_D0.2_Bi0_E15_B32" or "LSTM_L3_U64x64x32_D0.2_Bi0_E20_B32"
    tag_units = units if isinstance(units, int) else 'x'.join(map(str, units))
    tag = f"{layer_type}_L{num_layers}_U{tag_units}_D{dropout}_Bi{int(bidir)}_E{epochs}_B{batch}"
    model_path = os.path.join(MODEL_DIR, f"{tag}.keras")

    # Build the model from the factory; loss/optimiser kept constant for a fair comparison.
    model = build_dl_model(
        input_shape=input_shape,
        layer_type=layer_type,
        num_layers=num_layers,
        units=units,
        dropout=dropout,
        bidirectional=bidir,
        dense_units=1,
        loss="mse",
        optimizer="adam"
    )

    # Two callbacks:
    # - ModelCheckpoint keeps only the best weights (lowest val_loss).
    # - EarlyStopping prevents wasting epochs when validation stops improving.
    ckpt = ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss", verbose=0)
    es   = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0)

    # Train silently (verbose=0) because the grid prints a one-line summary per run below.
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch,
        callbacks=[ckpt, es],
        verbose=0
    )

    # Pull the quantities I care about:
    # - best validation loss (across epochs)
    # - final training loss (to eyeball generalisation gap)
    best_val = float(np.min(hist.history["val_loss"]))
    last_tr  = float(hist.history["loss"][-1])

    return {
        "tag": tag,
        "layer_type": layer_type,
        "num_layers": num_layers,
        # store units as JSON so lists survive the CSV round-trip cleanly
        "units": json.dumps(units if isinstance(units, list) else [units]),
        "dropout": dropout,
        "bidirectional": bidir,
        "epochs_ran": len(hist.history["loss"]),
        "batch_size": batch,
        "best_val_loss": best_val,
        "final_train_loss": last_tr,
        "model_path": model_path,
        "timestamp": int(time.time()),
    }


# --------- Run the grid and collect rows ---------
rows = []
for cfg in grid:
    try:
        res = run_one(cfg)
        rows.append(res)
        print(f"✔ {res['tag']} | best val_loss={res['best_val_loss']:.6f}")
    except Exception as e:
        # If something goes wrong with a particular config, I prefer to keep the loop going
        # and just record the failure in the console.
        print(f"✖ Failed on {cfg}: {e}")


# --------- Append results to CSV (create if missing) ---------
# I append to the same CSV so repeated runs build up a history I can sort/filter in Excel.
df = pd.DataFrame(rows)
if os.path.exists(EXP_CSV):
    prev = pd.read_csv(EXP_CSV)
    df = pd.concat([prev, df], ignore_index=True)
df.to_csv(EXP_CSV, index=False)

print(f"\nSaved experiment table → {EXP_CSV}")
