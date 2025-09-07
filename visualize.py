# File: visualize.py
# Purpose: two small helpers for Task C.3 —
#          (1) a candlestick chart that can compress n trading days into one candle,
#          (2) a rolling boxplot that shows how spread/volatility moves over time.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf


def _ensure_datetime_index(df):
    """
    Make sure the index is a DatetimeIndex. Both mplfinance and my own slicing-by-date
    assume a real timeline. If the CSV was read with dates as strings, I convert it here.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)  # safe conversion; keeps existing tz/format if present
    return df


def plot_candlestick_chart(
    df,
    ticker="STOCK",
    grouping_days=1,
    use_trading_days=True,
    mav=(5, 10),
    figsize=(12, 6),
    save_path=None
):
    """
    Draw a candlestick chart from OHLCV data with optional n-day grouping.

    Arguments:
      - df (DataFrame): must contain columns ['Open','High','Low','Close','Volume'].
                        Prices/volume should be *unscaled* so the chart is meaningful.
      - ticker (str): title label so the figure reads well in the report.
      - grouping_days (int): n ≥ 1. If n>1, I aggregate every n rows into a single candle.
                             Useful when there is too much data to read individual days.
      - use_trading_days (bool): True → group by *row count* (actual trading sessions).
                                 False → resample by calendar days (can introduce gaps).
      - mav (tuple|None): moving averages to overlay, e.g., (5,10). Use None to hide.
      - figsize (tuple): width, height in inches; bigger when plotting long histories.
      - save_path (str|None): if given, the figure is saved to this path after plotting.
    """
    df = _ensure_datetime_index(df)  # guarantee time-aware index for plotting

    # sanity: mplfinance expects exactly these columns; I prefer a clear error early
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"plot_candlestick_chart: missing columns {missing}. "
            f"Load with use_all_features=True so OHLCV are present."
        )

    df = df[required].copy()  # keep only what the chart needs; avoid stray columns confusing agg

    # Optional n-day grouping. Two routes: true trading-day blocks (row based) or calendar resample.
    if int(grouping_days) > 1:
        if use_trading_days:
            # --- trading-day grouping: build block IDs 0,0,0,1,1,1,... for n rows each
            gid = np.arange(len(df)) // int(grouping_days)
            g = df.assign(_gid=gid).groupby('_gid', sort=True)

            # For each block, compute OHLCV correctly:
            #   - Open  → first open in the block
            #   - High  → max across the block
            #   - Low   → min across the block
            #   - Close → last close in the block
            #   - Volume→ sum across the block
            open_   = g['Open'].first().rename('Open')
            high_   = g['High'].max().rename('High')
            low_    = g['Low'].min().rename('Low')
            close_  = g['Close'].last().rename('Close')
            volume_ = g['Volume'].sum().rename('Volume')

            grouped = pd.concat([open_, high_, low_, close_, volume_], axis=1)

            # Give the grouped candles a clean date index. Here I assign a simple daily range;
            # this keeps mplfinance happy without dragging in calendar gaps.
            start = pd.Timestamp.now().normalize()
            grouped.index = pd.date_range(start=start, periods=len(grouped), freq='D')
            df = grouped
        else:
            # --- calendar resample: aggregate every n *calendar* days. Handy if you want strict dates,
            #     but beware of weekends/holidays producing empty bins.
            rule = f"{int(grouping_days)}D"
            df = (
                df.resample(rule)
                  .agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'})
                  .dropna(how='any')  # if a bin is empty (e.g., all NaN), drop it
            )

    # Hand the prepared frame to mplfinance. I keep the style simple and the labels explicit.
    fig, _ = mpf.plot(
        df,
        type='candle',                 # candlesticks instead of OHLC bars/line
        volume=True,                   # add a volume subplot
        style='yahoo',                 # readable defaults (colors/axes)
        title=f"{ticker} Candles ({int(grouping_days)}-Day {'Trading' if use_trading_days else 'Calendar'} Grouping)",
        ylabel="Price",
        ylabel_lower="Volume",
        mav=mav,                       # moving averages drawn over the candles
        figsize=figsize,
        returnfig=True                 # request the figure handle so I can save it
    )

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)  # tidy borders, decent DPI for reports
    return fig


def plot_boxplot(
    df,
    price_column='Close',
    window_size=10,
    ticker='STOCK',
    figsize=(15,6),
    showfliers=False,
    save_path=None
):
    """
    Draw a rolling-window boxplot for a chosen price series (default: Close).

    Arguments:
      - df (DataFrame): time-indexed prices; unscaled preferred for meaningful axes.
      - price_column (str): which series to summarise ('Close' is typical).
      - window_size (int): number of trading rows per box (overlapping windows).
      - ticker (str): label for the title.
      - figsize (tuple): final size in inches; a bit wider helps when there are many boxes.
      - showfliers (bool): whether to draw outliers; I keep it False to reduce clutter.
      - save_path (str|None): optional file path to persist the figure.
    """
    df = _ensure_datetime_index(df)

    if price_column not in df.columns:
        raise ValueError(f"plot_boxplot: column '{price_column}' not in df.")

    prices = df[price_column].dropna().to_numpy()  # clean 1-D array of values

    box_data, labels = [], []  # collect window slices and matching end-date labels
    W = int(window_size)

    # Build overlapping windows: [0:W), [1:W+1), ...
    # Each window becomes a box; the label uses the window's end date for orientation.
    for i in range(len(prices) - W + 1):
        box_data.append(prices[i:i+W])
        labels.append(str(df.index[i+W-1].date()))

    # Standard matplotlib boxplot; patch_artist=True fills the boxes for readability.
    fig = plt.figure(figsize=figsize)
    plt.boxplot(box_data, patch_artist=True, showfliers=showfliers)
    plt.title(f"{ticker} {price_column} Boxplot (Rolling Window = {W} Trading Days)")
    plt.xlabel("Window End Date"); plt.ylabel(price_column)

    # Tick thinning: show ~10 evenly spaced labels so the x-axis stays readable.
    step = max(1, len(labels)//10)
    plt.xticks(ticks=np.arange(1, len(labels)+1, step), labels=labels[::step], rotation=45)

    # Light grid and tight layout so the figure drops neatly into the report.
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig
