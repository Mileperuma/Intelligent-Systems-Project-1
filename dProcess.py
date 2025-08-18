# File: dProcess.py
# Author: Matheesha
# Description: Handles data loading, processing, splitting and saving stuff
# Notes: This replaces the original bad data prep with something way more useful

import os
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# This function downloads stock data or loads it from local file (if saved before)
def load_data(ticker, start, end, reload_csv=False, use_all_features=True, save_dir='datasets'):
    os.makedirs(save_dir, exist_ok=True) # Just making sure the folder exists

    csv_path = os.path.join(save_dir, f'{ticker}_{start}_{end}.csv')
    scaler_path = os.path.join(save_dir, f'{ticker}_scaler.pkl')

    if not reload_csv and os.path.exists(csv_path): # If reload is False and we already have the CSV — just load it from disk
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        print(f"Loaded saved data from {csv_path}")
    else:
        # Download fresh data from Yahoo Finance
        df = yf.download(ticker, start=start, end=end)
        df.to_csv(csv_path)
        print(f"Downloaded and saved data to {csv_path}")

    # NaNs are common in finance data — let's just drop them for now
    df.dropna(inplace=True)

    # Choose features — either all (Open, High, Low, Close, Volume) or just Close
    # Choose features — either all (Open, High, Low, Close, Volume) or just Close
    if use_all_features:
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
    else:
        features = ['Close']

    # Drop all columns not in the feature list (in case AAPL, Adj Close etc sneak in)
    df = df[features]

    # Ensure all values are numeric (just in case some parsing failed)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)

    # Scale features to range [0, 1]
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=features, index=df.index)

    # Save the scaler too so we can use it again later during evaluation
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
        print(f"Saved scaler to {scaler_path}")

    return df_scaled, df, scaler


# Splits the dataset into training and testing sets
def split_data(df, train_ratio=0.8, split_by='date', random_seed=42):
    if split_by == 'date':
        # slicing the front part of the data as training (e.g. older data)
        # and use the rest (recent data) as testing
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
    elif split_by == 'random':
        train_df, test_df = train_test_split(df, train_size=train_ratio, random_state=random_seed, shuffle=True)
    else:
        # If user gives an invalid split method, throw a useful error
        raise ValueError("Invalid split method. Use 'date' or 'random'.")

    # Just print how much data went into train/test — helpful for debugging
    print(f"Data split into {len(train_df)} train rows and {len(test_df)} test rows")
    return train_df, test_df


# Converts raw stock data into sequential windowed format for LSTM input
def create_sequences(data, sequence_length):
    x = []  # input features
    y = []  # targets

    # If it's a DataFrame, convert to NumPy array. If it's already an array, just use it
    data_array = data.values if isinstance(data, pd.DataFrame) else data

    # Loop through the data and create sliding windows of length = sequence_length
    for i in range(len(data_array) - sequence_length):
        seq = data_array[i:i + sequence_length]  # Grab a chunk of data
        label = data_array[i + sequence_length][0]  # Target is the next 'Close' value (assumes it's the first col)

        # Sometimes the last sequence might not be full-length, so its always better double-check
        if len(seq) == sequence_length:
            x.append(seq)     # Add full sequence to feature list
            y.append(label)   # Add corresponding label

    # Convert lists into NumPy arrays so LSTM can train on them
    return np.array(x), np.array(y)

# This is like a wrapper to get all test data ready at once
def prepare_test_data(test_df, sequence_length=50):
    test_data = test_df.values
    x_test, y_test = create_sequences(test_data, sequence_length)
    print(f"Test sequences ready: {x_test.shape} {y_test.shape}")
    return x_test, y_test
