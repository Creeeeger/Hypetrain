import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.api import mixed_precision
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.src.models.sequential import Sequential
from keras.src.layers.regularization.dropout import Dropout
from keras.src.layers.core.dense import Dense
from keras.src.layers.rnn.lstm import LSTM
from keras.src.callbacks.early_stopping import EarlyStopping

tf.config.set_visible_devices([], 'GPU')  # Use only GPU
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)


# 1. Feature Engineering (Aligned with your Java indicators)
def create_features(df, window_size=60):
    """Create technical features from raw stock data"""
    df = df.copy()

    # Price transformations
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log1p(df['returns'])

    # Volatility
    df['volatility'] = df['returns'].rolling(20).std()

    # Momentum indicators
    df['sma_20'] = df['close'].rolling(20).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['rsi'] = compute_rsi(df['close'], 14)

    # Bollinger Bands
    df['upper_band'], df['lower_band'] = compute_bollinger_bands(df['close'], 20)

    # Target: 3% spike in next 15 minutes
    df['target'] = (df['close'].shift(-15) / df['close'] - 1 >= 0.03).astype(int)

    # Drop NA values created by rolling windows
    df.dropna(inplace=True)

    return df


def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_bollinger_bands(series, window):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return sma + 2 * std, sma - 2 * std


# 2. Data Preparation
def prepare_sequences(data, features, target, window_size=60):
    """Convert dataframe to LSTM sequences"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[features])

    x, y = [], []
    for i in range(window_size, len(data)):
        x.append(scaled_data[i - window_size:i])
        y.append(data[target].iloc[i])

    return np.array(x), np.array(y), scaler


# 3. LSTM Model Architecture
def build_spike_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape, recurrent_dropout=0.2),
        Dropout(0.3),
        LSTM(64, recurrent_dropout=0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')])
    return model


# 4. Training Pipeline
def train_spike_predictor(data_path):
    # Load and preprocess data
    df = pd.read_csv(data_path)
    df = create_features(df)

    # Feature selection
    features = ['close', 'returns', 'volatility', 'sma_20', 'macd', 'rsi', 'upper_band', 'lower_band']
    target = 'target'

    # Create sequences
    x, y, scaler = prepare_sequences(df, features, target)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Build and train model
    model = build_spike_model((X_train.shape[1], X_train.shape[2]))

    early_stop = EarlyStopping(monitor='val_precision',
                               patience=5,
                               mode='max',
                               restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=1,
        batch_size=256,
        validation_data=(X_test, y_test),
        callbacks=[early_stop]
    )

    # Save for production
    model.save('spike_predictor.keras')

    # Export scaler
    import joblib
    joblib.dump(scaler, 'scaler.pkl')

    return model, scaler


# Usage Example
if __name__ == "__main__":
    import tf2onnx

    # Train model
    model, scaler = train_spike_predictor('high_frequency_stocks.csv')

    model_proto, _ = tf2onnx.convert.from_keras(model)
    with open("spike_predictor.onnx", "wb") as f:
        f.write(model_proto.SerializeToString())