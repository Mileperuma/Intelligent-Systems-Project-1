# File: arima_utils.py
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def fit_arima(train_close, order=(5,1,0)):
    """
    Fit a simple ARIMA(p,d,q) on the *unscaled* training Close series.
    Returns the fitted results object.
    """
    model = ARIMA(train_close, order=order)
    return model.fit()

def forecast_rolling_matrix(result, test_len, horizon):
    long_fc = result.get_forecast(steps=test_len + horizon).predicted_mean.to_numpy()
    # Build sliding windows [i:i+H) across the long forecast
    mats = []
    for i in range(test_len - horizon + 1):
        mats.append(long_fc[i:i+horizon])
    return np.array(mats)
