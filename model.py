from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# This function builds and returns an LSTM model for time series prediction (e.g., stock prices)
def build_lstm_model(input_shape, lstm_units=64, dropout_rate=0.2):
    """
    Builds and returns an LSTM model with the given input shape.

    Parameters:
    - input_shape: shape of X_train (timesteps, features)
    - lstm_units: number of LSTM units per layer
    - dropout_rate: dropout to reduce overfitting

    Returns:
    - Compiled LSTM model
    """

    # We’re using Sequential API — just stack layers one after another
    model = Sequential()

    # First LSTM layer (with return_sequences=True to pass full sequence to next layer)
    model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))  # randomly turns off some neurons to reduce overfitting

    # Second LSTM layer (last one — doesn’t need to return sequence)
    model.add(LSTM(units=lstm_units))
    model.add(Dropout(dropout_rate))

    # Final output layer — just 1 neuron to predict the next value
    model.add(Dense(1))

    # Compile the model with Mean Squared Error loss and Adam optimizer
    # MSE is good for regression problems like this one
    model.compile(optimizer='adam', loss='mse')

    return model
