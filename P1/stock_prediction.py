import time, random
from collections import deque

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

# Seeds for repeatability
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)


def shuffle_in_unison(a, b):
    """Shuffle two arrays the same way (keeps (X, y) aligned)."""
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)


def _download_yf(ticker: str, tries: int = 3, sleep_s: float = 2.0) -> pd.DataFrame:
    """
    Try yfinance (two ways) then fall back to Stooq via pandas-datareader.
    Avoids Yahoo hiccups and campus network blocks when possible.
    """
    import yfinance as yf
    import pandas_datareader.data as pdr

    ticker = str(ticker)

    # --- Try 1: standard yf.download ---
    for _ in range(tries):
        df = yf.download(
            ticker, period="max", interval="1d",
            auto_adjust=False, progress=False, threads=False
        )
        if not df.empty and "Adj Close" in df.columns:
            df = df.rename(columns={
                "Adj Close": "adjclose",
                "Open": "open", "High": "high",
                "Low": "low", "Close": "close",
                "Volume": "volume",
            })
            return df
        time.sleep(sleep_s)

    # --- Try 2: Ticker().history ---
    for _ in range(tries):
        hist = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=False)
        if not hist.empty:
            hist = hist.rename(columns={
                "Adj Close": "adjclose",
                "Open": "open", "High": "high",
                "Low": "low", "Close": "close",
                "Volume": "volume",
            })
            return hist
        time.sleep(sleep_s)

    # --- Try 3: Stooq (often works when Yahoo is blocked) ---
    try:
        stq = pdr.DataReader(ticker, "stooq")  # newest->oldest
        if not stq.empty:
            stq = stq.sort_index()  # oldest->newest
            stq = stq.rename(columns={
                "Open": "open", "High": "high",
                "Low": "low", "Close": "close",
                "Volume": "volume",
            })
            stq["adjclose"] = stq["close"]  # Stooq has no Adj Close
            return stq[["open", "high", "low", "close", "volume", "adjclose"]]
    except Exception:
        pass

    raise ValueError(
        f"Could not fetch data for '{ticker}'. "
        "Try another ticker or another network (e.g., hotspot)."
    )


def load_data(
    ticker,
    n_steps=50,
    scale=True,
    shuffle=True,
    lookup_step=1,
    split_by_date=True,
    test_size=0.2,
    feature_columns=["adjclose", "volume", "open", "high", "low"],
):
    """
    Load, scale, window, and split data. Mirrors original P1 API/behavior.
    NOTE: For C1 reproducibility, we scale on the full dataset (like the tutorial).
    """
    # Accept string tickers or a pre-loaded DataFrame
    if isinstance(ticker, pd.DataFrame):
        df = ticker.copy()
    elif isinstance(ticker, (str, np.str_, bytes)):
        df = _download_yf(str(ticker))
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instance")

    result = {"df": df.copy()}

    # Ensure the requested features exist
    for col in feature_columns:
        if col not in df.columns:
            raise ValueError(f"Missing expected column '{col}' in dataframe.")

    # Add a 'date' column from the index if missing
    if "date" not in df.columns:
        df["date"] = df.index

    # --- Make features numeric & drop bad rows BEFORE scaling ---
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=feature_columns, inplace=True)

    # --- Scaling (safe shapes) ---
    column_scaler = {}
    if scale:
        for c in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            col = df[c].to_numpy()
            # Ensure exactly 2D (n,1)
            if col.ndim == 1:
                col = col.reshape(-1, 1)
            else:
                col = np.asarray(col).reshape(len(col), -1)[:, :1]
            df[c] = scaler.fit_transform(col).ravel()  # back to 1D
            column_scaler[c] = scaler
        result["column_scaler"] = column_scaler

    # Target is future scaled adjclose
    df["future"] = df["adjclose"].shift(-lookup_step)

    # Save last 'lookup_step' rows of features for forecasting
    last_sequence = np.array(df[feature_columns].tail(lookup_step))

    # Drop rows with NaNs from the shift
    df.dropna(inplace=True)

    # Build sequences
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns + ["date"]].values, df["future"].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # "last_sequence" for predicting unseen future
    last_sequence = list([s[: len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    result["last_sequence"] = last_sequence

    # Split into X and y
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    X, y = np.array(X), np.array(y)

    # Train/test split
    if split_by_date:
        train_samples = int((1 - test_size) * len(X))
        X_train, y_train = X[:train_samples], y[:train_samples]
        X_test, y_test = X[train_samples:], y[train_samples:]
        if shuffle:
            shuffle_in_unison(X_train, y_train)  # keep test in order
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)

    # Test dates (for plotting & CSV)
    test_dates = X_test[:, -1, -1]
    test_df = result["df"].loc[test_dates]
    test_df = test_df[~test_df.index.duplicated(keep="first")]

    # Remove 'date' column from tensors
    X_train = X_train[:, :, : len(feature_columns)].astype(np.float32)
    X_test = X_test[:, :, : len(feature_columns)].astype(np.float32)

    result.update({
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "test_df": test_df
    })
    return result


def create_model(
    sequence_length,
    n_features,
    units=256,
    cell=LSTM,
    n_layers=2,
    dropout=0.3,
    loss="mean_absolute_error",
    optimizer="rmsprop",
    bidirectional=False,
):
    """Build an LSTM stack with optional Bidirectional layers; same API as tutorial."""
    model = Sequential()
    for i in range(n_layers):
        return_seq = i != n_layers - 1
        if i == 0:
            layer = cell(units, return_sequences=return_seq, input_shape=(sequence_length, n_features))
        else:
            layer = cell(units, return_sequences=return_seq)
        model.add(Bidirectional(layer) if bidirectional else layer)
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model
