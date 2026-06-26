# File: dProcess.py
# Purpose: keep the data pipeline small and predictable: load OHLCV, scale it,
#          split if needed, and turn rows into the sliding windows our LSTM expects.
# Style: I explain decisions as I go, so future me (or a marker) can read intent
#        without opening another document.

import os          # folders and paths
import time        # timestamps for fallback filenames when a CSV is locked
import pickle      # persist the scaler so evaluation can invert the transform
import numpy as np # arrays for model input
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_data(ticker, start, end, reload_csv=False, use_all_features=True, save_dir='datasets'):
    """
    Download (or read cached) Yahoo Finance data, select features, scale to [0,1],
    and return both the scaled frame and the raw frame.

    Parameters (kept explicit so the call site reads like a sentence):
      - ticker (str): the stock symbol I want, e.g., "AAPL".
      - start (str | date-like): inclusive start date, e.g., "2010-01-01".
      - end   (str | date-like): exclusive end date for yfinance.
      - reload_csv (bool): False → reuse the cached CSV if present;
                           True  → ignore cache and re-download fresh data.
      - use_all_features (bool): True → keep OHLCV (Open, High, Low, Close, Volume);
                                 False → keep only Close (useful for quick baselines).
      - save_dir (str): folder where I cache the CSV and the scaler.

    Returns:
      - df_scaled (pd.DataFrame): scaled to [0,1], columns match the chosen features.
      - df        (pd.DataFrame): the raw (unscaled) frame, same columns as above.
      - scaler    (MinMaxScaler): fitted on the chosen columns so I can inverse-transform later.
    """
    os.makedirs(save_dir, exist_ok=True)  # create cache folder once; no error if it already exists

    # cache paths are deterministic so the same (ticker, start, end) resolves to the same file
    csv_path    = os.path.join(save_dir, f'{ticker}_{start}_{end}.csv')
    scaler_path = os.path.join(save_dir, f'{ticker}_scaler.pkl')

    # Either reuse the CSV (fast) or fetch anew (slow but reliable when I change dates/features)
    if (not reload_csv) and os.path.exists(csv_path):
        # index_col=0 → treat the first column as the index (it’s the Date in our CSV)
        # parse_dates=True → ensure the index is a proper DatetimeIndex for plotting later
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        print(f"Loaded saved data from {csv_path}")
    else:
        # Fresh download from Yahoo Finance; columns will include OHLCV (Adj Close may also appear)
        # yfinance uses [start, end) semantics; that’s fine for our use.
        df = yf.download(ticker, start=start, end=end)
        # Try to write the cache file. If it’s open in Excel, Windows will lock it;
        # I then fall back to a timestamped filename so the run can proceed.
        try:
            df.to_csv(csv_path)
            print(f"Downloaded and saved data to {csv_path}")
        except PermissionError:
            alt = os.path.join(save_dir, f'{ticker}_{start}_{end}_{int(time.time())}.csv')
            df.to_csv(alt)
            print(f"CSV was locked. Saved to {alt}")

    df.dropna(inplace=True)  # remove any incomplete rows early; plotting and scaling prefer clean data

    # Choose my feature set. For Task C.3 the visuals need OHLCV; the model can use all or just Close.
    cols_all = ['Open', 'High', 'Low', 'Close', 'Volume']
    features = cols_all if use_all_features else ['Close']

    # Keep only the chosen columns, coerce to numeric just in case, and drop any non-numeric leftovers
    df = df[features].apply(pd.to_numeric, errors='coerce').dropna()

    # Scale every chosen column to [0,1] so the LSTM doesn’t fight different magnitudes
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),   # fit on this exact set of columns, then transform them
        columns=features,
        index=df.index              # preserve dates so windows still align with time
    )

    # Persist the scaler so evaluation can inverse_transform model outputs back to price space
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
        print(f"Saved scaler to {scaler_path}")

    return df_scaled, df, scaler  # scaled for the model, raw for plots, scaler for reversing later


def split_data(df, train_ratio=0.8, split_by='date', random_seed=42):
    """
    Split a DataFrame into train/test, either by time order (head/tail) or randomly.

    Arguments:
      - df (pd.DataFrame): the frame I want to split.
      - train_ratio (float): proportion for training (e.g., 0.8 → 80% train, 20% test).
      - split_by (str): 'date' → first chunk is train, tail is test (respects chronology);
                        'random' → shuffle and split (useful for quick checks).
      - random_seed (int): fixed seed so the random split is repeatable when I need it.
    """
    if split_by == 'date':
        idx = int(len(df) * train_ratio)       # cut point
        train_df, test_df = df.iloc[:idx], df.iloc[idx:]  # chronological split
    elif split_by == 'random':
        # train_size controls the ratio; shuffle=True randomises rows; random_state makes it reproducible
        train_df, test_df = train_test_split(df, train_size=train_ratio, random_state=random_seed, shuffle=True)
    else:
        # I prefer a clear failure over a silent wrong split
        raise ValueError("Invalid split method. Use 'date' or 'random'.")

    print(f"Data split into {len(train_df)} train rows and {len(test_df)} test rows")
    return train_df, test_df

def create_sequences(data, sequence_length, target_index=0):
    """
    Turn rows into overlapping windows the LSTM understands.

    Arguments:
      - data (DataFrame or ndarray): features already in the order I want.
      - sequence_length (int): how many past timesteps to include in each window (e.g., 50).
      - target_index (int): which column the model should predict next (e.g., 3 → 'Close' in OHLCV).

    Returns:
      - X: shape (N, sequence_length, num_features)
      - y: shape (N,) — the next-step value from the chosen target column
    """
    x, y = [], []
    # If a DataFrame arrives, I pull its underlying array; if it’s already an array, leave it alone.
    arr = data.values if isinstance(data, pd.DataFrame) else data

    # Slide one step at a time: for each i, take [i : i+L) as input and the value at i+L as the target.
    for i in range(len(arr) - sequence_length):
        seq = arr[i:i + sequence_length]             # L rows of features → one training example
        tgt = arr[i + sequence_length][target_index] # the “next” value for the chosen target column
        x.append(seq)
        y.append(tgt)

    # Convert lists to numpy arrays so Keras can read shapes without guessing
    return np.array(x), np.array(y)

def create_multistep_sequences(data, sequence_length, target_index=0, horizon=5, step=1):
    """
    Build (X, Y) where:
      X shape = (N, sequence_length, num_features)
      Y shape = (N, horizon)  → next 'horizon' values of the target column

    Args:
      data : DataFrame or ndarray of scaled features
      sequence_length : int, input window length
      target_index : int, which column is the target (3 → 'Close' in OHLCV)
      horizon : int, number of future steps to predict (k)
      step : int, shift between consecutive windows (default 1)

    Returns:
      X, Y
    """
    arr = data.values if isinstance(data, pd.DataFrame) else data
    X, Y = [], []
    last_start = len(arr) - sequence_length - horizon + 1
    for i in range(0, max(0, last_start), step):
        X.append(arr[i:i+sequence_length])
        Y.append(arr[i+sequence_length:i+sequence_length+horizon, target_index])
    return np.array(X), np.array(Y)
