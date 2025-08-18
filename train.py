# This script handles the training process of our LSTM model
# It loads the data (already preprocessed), prepares sequences, trains the model, and saves it

import os
import numpy as np
from dProcess import load_data, create_sequences  # grabbing our custom data processing functions
from model import build_lstm_model               # the LSTM model builder we made earlier
from keras.callbacks import ModelCheckpoint      # for saving the best model during training

# ====== Configuration ======
TICKER = "AAPL"              # The stock we’re working with
START_DATE = "2010-01-01"    # Start date for data download
END_DATE = "2023-12-31"      # End date for data download
SEQ_LENGTH = 50              # Number of past days to look at for predicting the next
TARGET_COL = "Close"         # We’re only predicting the closing price
EPOCHS = 25                  # Total training rounds
BATCH_SIZE = 32              # Number of samples per training step
MODEL_SAVE_PATH = "models"   # Folder to save our trained model
MODEL_NAME = "lstm_model.keras"  # File name for the saved model

# ====== Load and prepare data ======
# We're calling our fancy data loader here — it'll grab from local or download if needed
train_df, test_df, scaler = load_data(ticker=TICKER, start=START_DATE, end=END_DATE)

# Convert dataframes into training-ready sequences
# X = sequences of past data, y = next-day target values
X_train, y_train = create_sequences(train_df.values, sequence_length=SEQ_LENGTH)
X_test, y_test = create_sequences(test_df.values, sequence_length=SEQ_LENGTH)

# ====== Model input shape ======
# LSTM layers need to know the shape of each input
# It should be (time_steps, num_features)
input_shape = (X_train.shape[1], X_train.shape[2])

# ====== Build the model ======
# This builds the LSTM model with our input shape — you can tweak it in model.py
model = build_lstm_model(input_shape)

# ====== Create save directory ======
# If the folder to store the model doesn't exist, create it
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

# ====== Save the best model during training ======
# We only want to keep the best model (lowest loss), not all of them
checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
    save_best_only=True,         # Only save if it's better than before
    monitor='loss',              # We're watching the training loss
    verbose=1                    # Print something when model is saved
)

# ====== Train the model ======
# We feed it the training data, validation data, and let it learn
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint]      # Auto-save best model during training
)

# Optional: Save training and validation loss values for future graphs or debugging
np.save(os.path.join(MODEL_SAVE_PATH, "train_loss.npy"), history.history["loss"])
np.save(os.path.join(MODEL_SAVE_PATH, "val_loss.npy"), history.history["val_loss"])

print("Training complete. Model saved to:", os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
