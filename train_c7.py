# train_c7.py
# Purpose: Train a Logistic Regression model for next-day price direction (classification).
# Compares performance of the model using all features vs. only non-sentiment features (baseline).

import pandas as pd, numpy as np, json, os
from sklearn.model_selection import TimeSeriesSplit # Not used, but often useful for proper time series CV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# --- Configuration ---
DATA_CSV = "sentiment/c7_dataset.csv" # The engineered dataset containing prices, technicals, and sentiment
OUT_DIR  = "plots_c7"; os.makedirs(OUT_DIR, exist_ok=True) # Output directory for metrics JSON and plots

# List of columns considered sentiment-derived features
SENT_COLS = ['pos','neg','neu','count','pos_lag1','neg_lag1','pos_3d','neg_3d']
# Baseline model will exclude these features.

def run_train(include_sentiment=True):
    """
    Load data, split, train a Logistic Regression classifier, and save performance metrics.

    Args:
        include_sentiment (bool): If True, use all features. If False, run the baseline
                                  by excluding sentiment columns.
    """
    # Load the combined feature dataset
    df = pd.read_csv(DATA_CSV, parse_dates=['date']).set_index('date')
    
    # Target variable: 'target_up' (1 if price goes up tomorrow, 0 if down/flat)
    y = df['target_up'].values

    # Feature set selection logic
    # Base columns are all non-target columns
    base_cols = [c for c in df.columns if c not in ['target_up','close_t+1']]
    
    if include_sentiment:
        # Full feature set (OHLCV, Technicals, Sentiment)
        X = df[base_cols].values
        tag = "with_sent"
    else:
        # Baseline feature set (OHLCV, Technicals, no Sentiment)
        base_wo_sent = [c for c in base_cols if c not in SENT_COLS]
        X = df[base_wo_sent].values
        tag = "baseline_no_sent"

    # Chronological split (ensures no look-ahead bias)
    n = len(df); split = int(n*0.8)
    # Train set is the first 80%, Test set is the final 20%
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    # Standardize features (mean=0, std=1) on the training set only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) # Fit and transform train data
    X_test  = scaler.transform(X_test)      # Transform test data using train stats

    # Initialize and train the classifier
    # 'balanced' adjusts weights to penalize misclassifying the smaller class more
    clf = LogisticRegression(max_iter=500, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Generate predictions on the held-out test set
    y_pred = clf.predict(X_test)

    # --- Metrics Calculation ---
    acc = accuracy_score(y_test, y_pred)
    # Calculate Precision, Recall, F1-Score (for both classes: Down=0, Up=1)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None) 
    # Calculate the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred).tolist() 

    # Bundle all metrics into a dictionary
    metrics = {
        "tag": tag,
        "accuracy": float(acc),
        "precision_down": float(p[0]),
        "recall_down": float(r[0]),
        "f1_down": float(f1[0]),
        "precision_up": float(p[1]),
        "recall_up": float(r[1]),
        "f1_up": float(f1[1]),
        "confusion_matrix": cm,
    }

    # Save metrics as a JSON file for the evaluation script
    with open(os.path.join(OUT_DIR, f"c7_metrics_{tag}.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    return metrics

if __name__ == "__main__":
    # Run the model with sentiment features
    run_train(include_sentiment=True)
    # Run the baseline model without sentiment features
    run_train(include_sentiment=False)