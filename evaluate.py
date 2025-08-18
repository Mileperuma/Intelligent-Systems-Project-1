import numpy as np
import matplotlib.pyplot as plt
from dProcess import load_data, create_sequences
from keras.models import load_model
# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
TICKER = "AAPL"
START_DATE = "2010-01-01"
END_DATE = "2023-12-31"
SEQUENCE_LENGTH = 50
TARGET_COLUMN = "Close"

# -------------------------------------------------------------
# Load preprocessed data
# -------------------------------------------------------------
# We only care about test set here, so we ignore training data
# reload_csv=True makes sure it uses saved CSV if available
_, test_df, scaler = load_data(
    ticker=TICKER,
    start=START_DATE,
    end=END_DATE,
    reload_csv=True
)

# -------------------------------------------------------------
# Prepare sequences for prediction
# -------------------------------------------------------------
# This builds test input-output pairs using a sliding window
X_test, y_test = create_sequences(test_df, SEQUENCE_LENGTH)

# -------------------------------------------------------------
# Output the shape of test sets
# -------------------------------------------------------------
# X_test: (num_sequences, sequence_length, num_features)
# y_test: (num_sequences,)

# Load the trained model
model = load_model("models/lstm_model.keras")

# Predict using the test sequences
# Run the trained model to get predictions for the test data
predicted = model.predict(X_test)

# Since we scaled the data earlier, now we need to reverse that scaling
# We're padding the predictions with zeros to match original feature count (for inverse_transform)
predicted = scaler.inverse_transform(
    np.concatenate([predicted, np.zeros((predicted.shape[0], scaler.n_features_in_ - 1))], axis=1)
)[:, 0]  # Only keep the first column (usually the 'Close' price)

# Same inverse scaling for the actual y_test values so we can compare both on original scale
y_test_rescaled = scaler.inverse_transform(
    np.concatenate([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaler.n_features_in_ - 1))], axis=1)
)[:, 0]

# Print the shape of the test data just for verification/logging
print("Test sequences ready:", X_test.shape, y_test.shape)

# Visualising the predictions vs actual values
plt.figure(figsize=(14, 5))  # Set the plot size
plt.plot(y_test, label='Actual')  # Plot the actual stock prices
plt.plot(predicted, label='Predicted')  # Plot the predicted stock prices
plt.title(f'{TICKER} Stock Price Prediction')  # Title of the graph
plt.xlabel('Time')  # X-axis label
plt.ylabel('Price')  # Y-axis label
plt.legend()  # Show legend for 'Actual' and 'Predicted'
plt.show()  # Finally, display the plot
