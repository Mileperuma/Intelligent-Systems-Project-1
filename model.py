# File: model.py
# Purpose: build RNN-family models from a compact config: {layer_type, num_layers, units, dropout}

from keras.models import Sequential
from keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout, Bidirectional

def _pick_layer(layer_type: str):
    """Map a short name to the correct Keras recurrent layer class."""
    lt = layer_type.strip().upper()
    if lt == "LSTM":
        return LSTM
    if lt == "GRU":
        return GRU
    if lt in ("RNN", "SIMPLERNN"):
        return SimpleRNN
    raise ValueError(f"Unsupported layer_type: {layer_type}. Use LSTM, GRU, or RNN.")

def build_dl_model(
    input_shape,
    layer_type="LSTM",
    num_layers=2,
    units=64,
    dropout=0.2,
    bidirectional=False,
    dense_units=1,
    loss="mse",
    optimizer="adam"
):
    """
    Build a recurrent model from config.

    Args:
      input_shape : tuple, (timesteps, features) — e.g., (50, 5)
      layer_type  : 'LSTM' | 'GRU' | 'RNN' (SimpleRNN)
      num_layers  : int >= 1, how many recurrent layers to stack
      units       : int or list[int], hidden size(s) per recurrent layer
      dropout     : float in [0,1), dropout after each recurrent layer
      bidirectional: bool, wrap recurrent layers with Bidirectional if True
      dense_units : int, size of final Dense output (1 for next-step regression)
      loss        : keras loss string (default 'mse')
      optimizer   : keras optimizer string (default 'adam')

    Returns:
      Compiled keras.Model ready to fit().
    """
    RNN = _pick_layer(layer_type)
    model = Sequential()

    # normalise units to a list of length num_layers
    if isinstance(units, int):
        units_list = [units] * num_layers
    else:
        assert len(units) == num_layers, "len(units) must equal num_layers"
        units_list = list(units)

    for i, u in enumerate(units_list):
        # By default, all but the last recurrent layer must return sequences.
        return_seq = (i < num_layers - 1)
        layer = RNN(u, return_sequences=return_seq, input_shape=input_shape if i == 0 else None)
        if bidirectional:
            layer = Bidirectional(layer)

        model.add(layer)
        if dropout and dropout > 0:
            model.add(Dropout(dropout))

    # Final regression head
    model.add(Dense(dense_units))
    model.compile(optimizer=optimizer, loss=loss)
    return model


# Backward-compatible helper (used in v0.3); keeps your older scripts working.
def build_lstm_model(input_shape, lstm_units=64, dropout_rate=0.2):
    model = Sequential()
    model.add(LSTM(lstm_units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model
