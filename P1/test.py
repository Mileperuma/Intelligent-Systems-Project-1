import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stock_prediction import create_model, load_data
from parameters import *

def plot_graph(test_df):
    plt.plot(test_df[f'true_adjclose_{LOOKUP_STEP}'], c='b')
    plt.plot(test_df[f'adjclose_{LOOKUP_STEP}'], c='r')
    plt.xlabel("Days"); plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()

def get_final_df(model, data):
    buy_profit  = lambda c, pf, tf: (tf - c) if pf > c else 0
    sell_profit = lambda c, pf, tf: (c - tf) if pf < c else 0
    X_test, y_test = data["X_test"], data["y_test"]
    y_pred = model.predict(X_test)
    if SCALE:
        y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    test_df = data["test_df"].copy()
    test_df[f"adjclose_{LOOKUP_STEP}"] = y_pred
    test_df[f"true_adjclose_{LOOKUP_STEP}"] = y_test
    test_df.sort_index(inplace=True)
    test_df["buy_profit"]  = list(map(buy_profit,  test_df["adjclose"], test_df[f"adjclose_{LOOKUP_STEP}"], test_df[f"true_adjclose_{LOOKUP_STEP}"]))
    test_df["sell_profit"] = list(map(sell_profit, test_df["adjclose"], test_df[f"adjclose_{LOOKUP_STEP}"], test_df[f"true_adjclose_{LOOKUP_STEP}"]))
    return test_df

def predict(model, data):
    last = np.expand_dims(data["last_sequence"][-N_STEPS:], axis=0)
    pred = model.predict(last)
    return data["column_scaler"]["adjclose"].inverse_transform(pred)[0][0] if SCALE else pred[0][0]

# ----- Prefer cached CSV (avoids network issues) -----
os.makedirs("data", exist_ok=True)
csvs = sorted(glob.glob(os.path.join("data", f"{ticker}_*.csv")), key=os.path.getmtime)
if csvs:
    df = pd.read_csv(csvs[-1], index_col=0, parse_dates=[0])

    # Normalize column names to what the model expects
    rename_map = {
        "Adj Close": "adjclose",
        "Open": "open", "High": "high",
        "Low": "low", "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    # Keep only features; force numeric; drop bad rows
    keep = ["adjclose", "volume", "open", "high", "low"]
    df = df[[c for c in keep if c in df.columns]].copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=df.columns)

    data = load_data(df, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE,
                     shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                     feature_columns=FEATURE_COLUMNS)
else:
    data = load_data(ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE,
                     shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                     feature_columns=FEATURE_COLUMNS)

# ----- Build model & load newest weights for this ticker -----
model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS,
                     cell=CELL, n_layers=N_LAYERS, dropout=DROPOUT,
                     optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

os.makedirs("results", exist_ok=True)
weight_candidates = glob.glob(os.path.join("results", f"*_{ticker}-*.weights.h5"))
if not weight_candidates:
    raise FileNotFoundError("No weights found in 'results/'. Train first (python train.py).")
weights_path = max(weight_candidates, key=os.path.getmtime)
print(f"Loading weights: {os.path.basename(weights_path)}")
model.load_weights(weights_path)

# ----- Evaluate, report, plot, save CSV -----
loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
mae_real = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0] if SCALE else mae

final_df = get_final_df(model, data)
future_price = predict(model, data)

accuracy = (len(final_df[final_df['sell_profit'] > 0]) + len(final_df[final_df['buy_profit'] > 0])) / len(final_df)
total_buy_profit  = final_df["buy_profit"].sum()
total_sell_profit = final_df["sell_profit"].sum()
total_profit = total_buy_profit + total_sell_profit
profit_per_trade = total_profit / len(final_df)

print(f"Future price after {LOOKUP_STEP} days: {future_price:.2f}$")
print("Loss:", loss)
print("Mean Absolute Error:", mae_real)
print("Accuracy:", accuracy)
print("Total buy profit:", total_buy_profit)
print("Total sell profit:", total_sell_profit)
print("Total profit:", total_profit)
print("Profit per trade:", profit_per_trade)

plot_graph(final_df)

os.makedirs("csv-results", exist_ok=True)
out_csv = os.path.join("csv-results", os.path.basename(weights_path).replace(".weights.h5", ".csv"))
final_df.to_csv(out_csv)
print(f"Saved: {out_csv}")

# After the plotting code (where you do plt.plot...)
plt.savefig("prediction_plot.png")  # Saves the plot as an image
