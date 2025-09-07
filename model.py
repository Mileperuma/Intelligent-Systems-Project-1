from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# build_lstm_model:
# a small helper that constructs and compiles an LSTM network tailored for sequence prediction.
# I keep the defaults simple but parameterised so I can tweak units or dropout if needed.
def build_lstm_model(input_shape, lstm_units=64, dropout_rate=0.2):
    """
    Arguments:
      - input_shape (tuple): (timesteps, features). For example, (50,5) means
                             50 days of history, 5 features each (OHLCV).
      - lstm_units (int): number of hidden units in each LSTM layer. More units
                          let the network capture richer temporal patterns but
                          also make it heavier to train.
      - dropout_rate (float): proportion of units randomly dropped during training.
                              A small guard against overfitting (e.g., 0.2 = 20%).

    Returns:
      - a compiled Keras Sequential model, ready to .fit() on windowed data.
    """
    model = Sequential()  # simple stack of layers; no fancy branching required here

    # First LSTM: return_sequences=True so the whole sequence is passed on to the next LSTM.
    model.add(LSTM(lstm_units, return_sequences=True, input_shape=input_shape))
    # Dropout after each recurrent layer: intentionally forget some connections to improve generalisation.
    model.add(Dropout(dropout_rate))

    # Second LSTM: the final recurrent layer, outputs a vector (no need to return sequences here).
    model.add(LSTM(lstm_units))
    model.add(Dropout(dropout_rate))

    # Dense(1): a single linear neuron — the next-step prediction for the chosen target column.
    model.add(Dense(1))

    # Compile the stack: 'adam' is a robust default optimiser,
    # 'mse' is a good fit for regression problems like predicting stock prices.
    model.compile(optimizer='adam', loss='mse')

    return model
