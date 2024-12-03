import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import os
import random

# Set random seed for reproducibility
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

# Load and preprocess data
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    return data

# Create sequences for LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length)].values
        y = data.iloc[i + seq_length]['ADA_USDT_close']
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Load data
data = load_and_preprocess_data('data_prepared/merged_crypto_data.csv')

# Select target column and features
features = data.columns

# Normalize data
scaler = RobustScaler()
data[features] = scaler.fit_transform(data[features])

# Set sequence 31 days
seq_length = 14 * 24  # 31 days * 24 hours
X, y = create_sequences(data[features], seq_length)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model with Bidirectional LSTM and Batch Normalization
model = Sequential([
    Bidirectional(LSTM(400, return_sequences=True, input_shape=(seq_length, len(features)))),
    BatchNormalization(),
    Dropout(0.3),
    Bidirectional(LSTM(200, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.3),
    Bidirectional(LSTM(100)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(50, activation='relu'),
    BatchNormalization(),
    Dense(1)
])

# Use a more sophisticated optimizer and loss function
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='huber'  # Huber loss is more robust to outliers
)

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    min_delta=1e-4
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Save the model
model.save('trained_model.h5')

# Evaluate the model
y_pred = model.predict(X_test)
y_test_actual = y_test

mse = mean_squared_error(y_test_actual, y_pred)
mae = mean_absolute_error(y_test_actual, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

# Plot predicted vs actual prices for random 1-year slices
num_plots = 5
slice_length = 365  # 1 year

for i in range(num_plots):
    start_idx = np.random.randint(0, len(y_test_actual) - slice_length)
    end_idx = start_idx + slice_length
    
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[start_idx:end_idx], y_pred[start_idx:end_idx], label='Predicted Prices', alpha=0.5)
    plt.plot(data.index[start_idx:end_idx], y_test_actual[start_idx:end_idx], label='Actual Prices')
    plt.title(f'Actual vs Predicted Prices (Slice {i+1})')
    plt.xlabel('Date')
    plt.ylabel('ADA Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.savefig(f'predicted_vs_actual_prices_slice_{i+1}.png')
    plt.close()