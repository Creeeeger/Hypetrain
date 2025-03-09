import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tf2onnx
from imblearn.over_sampling import SMOTE
from keras import Input, Model
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import LSTM, Dense, Dropout, BatchNormalization, MaxPooling1D, Conv1D
from keras.src.metrics import Precision, Recall, AUC
from keras.src.regularizers import regularizers
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# 1. Feature Engineering
def create_features(data_of_csv):
    data_of_csv = data_of_csv.copy()
    data_of_csv['returns'] = data_of_csv['close'].pct_change(fill_method=None)

    data_of_csv['short_sma'] = calculate_sma(data_of_csv['close'], 9)
    data_of_csv['long_sma'] = calculate_sma(data_of_csv['close'], 21)

    bullish_crossover = (
            (data_of_csv['short_sma'] > data_of_csv['long_sma']) &
            (data_of_csv['short_sma'].shift(1) <= data_of_csv['long_sma'].shift(1))
    )

    bearish_crossover = (
            (data_of_csv['short_sma'] < data_of_csv['long_sma']) &
            (data_of_csv['short_sma'].shift(1) >= data_of_csv['long_sma'].shift(1))
    )

    # Initialize crossover column with 0
    data_of_csv['sma_crossover'] = 0

    # Mark crossover events
    data_of_csv.loc[bullish_crossover, 'sma_crossover'] = 1
    data_of_csv.loc[bearish_crossover, 'sma_crossover'] = -1

    # Forward fill to maintain state between crossovers
    data_of_csv['sma_crossover'] = (
        data_of_csv['sma_crossover']
        .replace(0, np.nan)  # Replace 0s with NaN for forward filling
        .ffill()  # Propagate last known state forward
        .fillna(0)  # Fill initial NaNs (pre-first crossover) with 0
        .astype(int)
    )

    data_of_csv['trix'] = calculate_trix(data_of_csv, period=5)

    data_of_csv['roc'] = calculate_roc(data_of_csv, period=20)

    data_of_csv['bollinger_bands'] = calculate_bollinger_bands(data_of_csv, period=20)

    data_of_csv['cumulative_spike'] = calculate_cumulative_spike(data_of_csv, period=10, threshold=0.35)

    data_of_csv['cumulative_change'] = calculate_cumulative_change(data_of_csv, last_change_length=8)

    data_of_csv['keltner_breakout'] = calculate_keltner_breakout(data_of_csv, ema_period=12, atr_period=10,
                                                                 multiplier=0.3)

    data_of_csv['elder_ray_index'] = calculate_elder_ray_index(data_of_csv, ema_period=12)

    data_of_csv['target'] = data_of_csv['target'].astype(int)

    data_of_csv.dropna(inplace=True)

    return data_of_csv


def calculate_sma(series, period):
    return series.rolling(window=period).mean()


def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calculate_roc(data_of_csv, period):
    return data_of_csv['close'].pct_change(periods=period, fill_method=None) * 100


def calculate_trix(data_of_csv, period):
    # Calculate Triple EMA
    ema1 = calculate_ema(data_of_csv['close'], period)
    ema2 = calculate_ema(ema1, period)
    ema3 = calculate_ema(ema2, period)

    # TRIX: Percentage rate of change
    trix = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    return trix


def calculate_bollinger_bands(data_of_csv, period):
    if len(data_of_csv) < period:
        return np.full((len(data_of_csv), 4), np.nan)

    # Precompute rolling metrics in one pass
    close = data_of_csv['close'].values
    rolling_mean = np.empty(len(close))
    rolling_std = np.empty(len(close))

    for i in range(len(close)):
        start = max(0, i - period + 1)
        window = close[start:i + 1]
        rolling_mean[i] = window.mean()
        rolling_std[i] = window.std(ddof=0) if len(window) > 1 else 0

    upper = rolling_mean + 2 * rolling_std
    lower = rolling_mean - 2 * rolling_std
    bandwidth = (upper - lower) / np.where(rolling_mean != 0, rolling_mean, np.nan)

    return bandwidth


def calculate_cumulative_spike(data, period, threshold):
    # Calculate returns if not already present
    if 'returns' not in data:
        data['returns'] = data['close'].pct_change()

    # Use logarithmic returns for numerical stability
    log_returns = np.log1p(data['returns'])

    # Calculate rolling cumulative return using sum of logs
    cumulative_return = (
        log_returns
        .rolling(window=period, min_periods=period)
        .sum()
        .shift(1)  # Look back at previous period window
        .pipe(lambda x: np.exp(x) - 1)  # Convert back to linear scale
        .mul(100)  # Convert to percentage
    )

    # Create spike indicator
    return (cumulative_return >= threshold).fillna(0).astype(int)


def calculate_cumulative_change(data, last_change_length):
    # Calculate returns if missing
    if 'returns' not in data:
        data['returns'] = data['close'].pct_change()

    # Use logarithmic returns for numerical stability
    log_returns = np.log1p(data['returns'])

    # Calculate rolling sum of log returns
    cumulative_log = (
        log_returns
        .rolling(window=last_change_length, min_periods=last_change_length)
        .sum()
        .shift(1)  # Align with previous window
    )

    # Convert back to linear scale and format
    return (
            np.exp(cumulative_log) - 1
    ).mul(100).fillna(0).astype(float)


def calculate_keltner_breakout(data, ema_period, atr_period, multiplier):
    # Precompute EMA and ATR
    data['ema'] = data['close'].ewm(span=ema_period, adjust=False).mean()
    data['atr'] = calculate_atr(data, atr_period)

    # Calculate upper band
    upper_band = data['ema'] + (multiplier * data['atr'])

    # Determine breakouts
    breakouts = (data['close'] > upper_band).astype(int)

    # Cleanup temporary columns
    data.drop(columns=['ema', 'atr'], inplace=True)

    return breakouts


def calculate_elder_ray_index(data, ema_period):
    ema = data['close'].ewm(span=ema_period, adjust=False).mean()
    return data['close'] - ema


def calculate_atr(data_of_csv, period):
    high_low = data_of_csv['high'] - data_of_csv['low']
    high_close = np.abs(data_of_csv['high'] - data_of_csv['close'].shift())
    low_close = np.abs(data_of_csv['low'] - data_of_csv['close'].shift())

    true_range = np.maximum(high_low, high_close, low_close)
    atr = true_range.rolling(window=period).mean()
    return atr


# 2. Dataset preparation
def prepare_sequences(data, features, window_size):
    # Identify numeric features
    numeric_features = data[features].select_dtypes(include=['float64', 'int64']).columns

    # Check for NaN values and drop rows with missing values
    if data[numeric_features].isna().any().any():
        data = data.dropna(subset=numeric_features)

    # Scale all numeric features using MinMaxScaler for consistency (instead of low-range/high-range distinction)
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', MinMaxScaler(), numeric_features)
        ]
    )

    # Apply the preprocessor to the data
    scaled_data = preprocessor.fit_transform(data[features])

    label = data['target'].values[window_size:]

    feature = []
    for i in range(window_size, len(data)):
        feature.append(scaled_data[i - window_size:i])

    return np.array(feature), np.array(label), preprocessor


# 3. Model architecture construction
def build_model(input_shape):
    l2s = regularizers.L2(0.001)
    inputs = Input(shape=input_shape)

    # ========== CNN Layers ==========
    # First Conv Block
    x = Conv1D(64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2s)(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)

    # Second Conv Block
    x = Conv1D(128, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2s)(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)

    # ========== LSTM Layer ==========
    x = LSTM(64, return_sequences=False, kernel_regularizer=l2s, recurrent_activation="tanh")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # ========== Dense Layers ==========
    x = Dense(64, activation='relu', kernel_regularizer=l2s)(x)
    x = Dropout(0.3)(x)

    outputs = Dense(1, activation='sigmoid', dtype='float32')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy',
                           Precision(name='precision'),
                           Recall(name='recall'),
                           AUC(name='auc')])
    return model


# 4. Trainings loop
def train_spike_predictor(data_path):
    # read the data of the CSV file and skip bad lines to prevent errors
    data_of_csv = pd.read_csv(data_path, on_bad_lines='skip')

    # Engineer the features
    data_of_csv = create_features(data_of_csv)

    # feature categories from Java feature creator
    features_list = [
        'sma_crossover', 'trix',
        'roc',
        'bollinger_bands',
        'cumulative_spike', 'cumulative_change',
        'keltner_breakout', 'elder_ray_index'
    ]

    # Scale the data to values between 0 and 1
    features, labels, scaler = prepare_sequences(data_of_csv, features_list, len(features_list))

    # Split the data to test_size X for testing and 1-X for training
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20)

    # Build and define the architecture of the Network for training
    model = build_model((features_train.shape[1], features_train.shape[2]))

    # Define callbacks
    early_stop = EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True, min_delta=0.005)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)

    # Reshape data for SMOTE
    nsamples, window_size, n_features = features_train.shape
    features_train_2d = features_train.reshape(nsamples, window_size * n_features)

    # Apply SMOTE to balance classes
    smote = SMOTE(random_state=42)
    features_train_res_2d, labels_train_res = smote.fit_resample(features_train_2d, labels_train)

    # Reshape back to 3D for model input
    features_train_res = features_train_res_2d.reshape(-1, window_size, n_features)

    # Update class weight calculation using resampled data
    print("Resampled class distribution:", np.bincount(labels_train_res.flatten()))
    print("Class distribution:", np.bincount(labels_train.flatten()))

    # training process of the model CPU has in my case better performance than GPU
    # Explanation: tasks are moved around from cpu to gpu vice versa. Sometimes tasks are split between both which
    # introduces the constant of latency between communication which is 20ms per step vs 10 ms
    # since the architecture and the wide variety of components of this network don't allow
    # execution on one EP only efficient TensorFlow splits the process up but that isn't effective
    # hybrid models with assigning certain tasks forceful to EPs did improve the performance by 25% to 15 ms.
    with tf.device('/CPU:0'):
        model.fit(
            features_train_res, labels_train_res,
            epochs=256,
            batch_size=128,
            validation_data=(features_test, labels_test),
            callbacks=[early_stop, reduce_lr],
        )

    # Convert and save the ONNX model directly & define the input signature dynamically based on training data shape
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=[
        tf.TensorSpec([None, *features_train.shape[1:]], tf.float32)])
    with open("spike_predictor.onnx", "wb") as f:
        f.write(model_proto.SerializeToString())


if __name__ == "__main__":
    # Disable GPU to get 50% faster training (see explanation before training)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    tf.config.set_visible_devices([], 'GPU')

    train_spike_predictor('highFrequencyStocks.csv')  # Main function for training
    print("Training done!")