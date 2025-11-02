# build_features_c7.py
# Purpose: Combine raw OHLCV price data with calculated technical indicators and daily FinBERT sentiment scores
#          to create the final, integrated dataset for a classification model.

import sys, os
# Adjust the system path to allow importing utility modules from a parent directory (e.g., dProcess)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dProcess import load_data
import pandas as pd
import numpy as np


def technicals_from_raw(raw_df):
    """Calculates technical indicators for the DataFrame."""
    df = raw_df.copy()
    # Calculate the 1-day percentage return (a volatility feature)
    df['return_1d'] = df['Close'].pct_change()

    # RSI-14 (Relative Strength Index: a key momentum indicator)
    df['rsi_14'] = df['Close'].pct_change().rolling(14).apply(
        # The core RSI formula implemented via a rolling custom function
        lambda x: 100 - (100/(1 + (x[x > 0].sum() / (1e-9 + -x[x < 0].sum())))),
        raw=True
    )

    # 5-day moving avg on volume (Volume Simple Moving Average to smooth noise)
    df['vol_ma5'] = df['Volume'].rolling(5).mean()
    return df


def make_dataset(ticker="AAPL",
                 start="2010-01-01", end="2023-12-31",
                 sent_csv="sentiment/daily_sentiment.csv",
                 out_csv="sentiment/c7_dataset.csv"):
    """
    Main function to load, process, merge data, and create the classification target.
    """

    # Load main OHLCV (Open, High, Low, Close, Volume) data
    scaled, raw, _ = load_data(
        ticker, start, end,
        reload_csv=False,
        use_all_features=True
    )

    prices = raw.copy()
    # Ensure index is a date-only object for consistent joining
    prices.index = pd.to_datetime(prices.index).date

    # ----- Technical features -----
    tech = technicals_from_raw(prices)

    # ----- Sentiment -----
    sent = pd.read_csv(sent_csv)
    sent['date'] = pd.to_datetime(sent['date']).dt.date
    # Set the date as the index and sort chronologically
    sent = sent.set_index('date').sort_index()

    # lag sentiment features (use previous day's sentiment to predict today)
    sent['pos_lag1'] = sent['pos'].shift(1) # Positive score from t-1
    sent['neg_lag1'] = sent['neg'].shift(1) # Negative score from t-1
    sent['pos_3d'] = sent['pos'].rolling(3).mean() # 3-day rolling mean of positive score
    sent['neg_3d'] = sent['neg'].rolling(3).mean() # 3-day rolling mean of negative score

    # ----- Merge -----
    # Left join price/technical data with sentiment data on the date index
    df = tech.join(sent, how='left')

    # ---- Fill sentiment gaps → neutral when no news ----
    sent_cols = ['pos', 'neg', 'neu', 'count', 'pos_lag1', 'neg_lag1', 'pos_3d', 'neg_3d']
    for c in sent_cols:
        if c in df.columns:
            # Days with missing sentiment (no news) are set to 0.0
            df[c] = df[c].fillna(0.0)

    # ---- Forward-fill technical indicators ----
    # Fills NaNs (like the first 13 days of RSI) with the last available value
    df = df.fillna(method='ffill')

    # ---- Target ----
    # 1. Price at the next trading day (t+1)
    df['close_t+1'] = prices['Close'].shift(-1)
    # 2. Binary Classification Target: 1 if Close goes UP tomorrow, 0 if Down/Flat
    df['target_up'] = (df['close_t+1'] > prices['Close']).astype(int)

    # drop last row where target is undefined (as shift(-1) creates a NaN at the end)
    df = df.iloc[:-1]

    # ---- Final cleanup (NO NaN remains) ----
    # Replace any extreme infinite values with NaN, then fill remaining NaNs with 0
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    # ----- Save -----
    df.to_csv(out_csv, index_label='date')
    print("Saved:", out_csv)


if __name__ == "__main__":
    make_dataset()