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

    data_of_csv['macd_line'] = calculate_ema(data_of_csv['close'], 6) - calculate_ema(data_of_csv['close'], 13)
    data_of_csv['signal_line'] = calculate_ema(data_of_csv['macd_line'], 5)
    data_of_csv['macd_histogram'] = data_of_csv['macd_line'] - data_of_csv['signal_line']

    data_of_csv['trix'] = calculate_trix(data_of_csv, period=5)

    data_of_csv['rsi'] = calculate_rsi(data_of_csv, period=15)

    data_of_csv['roc'] = calculate_roc(data_of_csv, period=20)

    data_of_csv['momentum'] = calculate_momentum(data_of_csv, period=10)

    data_of_csv['cmo'] = calculate_cmo(data_of_csv, period=20)

    data_of_csv['bollinger_bands'] = calculate_bollinger_bands(data_of_csv, period=20)

    data_of_csv['positive_closes'] = calculate_positive_closes(data_of_csv)

    data_of_csv['higher_highs'] = calculate_higher_highs(data_of_csv, min_consecutive=3)

    data_of_csv['trendline_breakout'] = is_trendline_breakout(data_of_csv, lookback=20)

    data_of_csv['cumulative_spike'] = calculate_cumulative_spike(data_of_csv, period=10, threshold=0.55)

    data_of_csv['cumulative_change'] = calculate_cumulative_change(data_of_csv)

    data_of_csv['parabolic_sar_bullish'] = is_parabolic_sar_bullish(data_of_csv, period=20, acceleration=0.01)

    data_of_csv['keltner_breakout'] = calculate_keltner_breakout(data_of_csv)

    data_of_csv['elder_ray_index'] = calculate_elder_ray_index(data_of_csv)

    data_of_csv['atr'] = calculate_atr(data_of_csv, period=20)

    data_of_csv['target'] = data_of_csv['target'].astype(int)

    data_of_csv.dropna(inplace=True)

    return data_of_csv


def calculate_sma(series, period):
    return series.rolling(window=period).mean()


def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calculate_cmo(data_of_csv, period):
    delta = data_of_csv['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).sum()
    loss = -delta.where(delta < 0, 0).rolling(window=period).sum()
    cmo = (gain - loss) / (gain + loss) * 100
    return cmo


def calculate_momentum(data_of_csv, period):
    momentum = data_of_csv['close'].diff(period)
    return momentum


def calculate_roc(data_of_csv, period):
    return data_of_csv['close'].pct_change(periods=period, fill_method=None) * 100


def calculate_rsi(data_of_csv, period):
    delta = data_of_csv['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


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


def calculate_positive_closes(data, dip_tolerance=0.2):
    if 'returns' not in data:
        data['returns'] = data['close'].pct_change()

    returns_pct = data['returns'] * 100
    n = len(data)
    counts = np.zeros(n, dtype=int)
    current_streak = 0

    for i in range(1, n):
        prev_return = returns_pct.iloc[i - 1]

        # Reset streak if previous period had a dip
        if prev_return < -dip_tolerance:
            current_streak = 0
        # Increment streak if previous period was positive
        elif prev_return > 0:
            current_streak += 1
        # Reset for neutral returns (0 or between 0 and -tolerance)
        else:
            current_streak = 0

        counts[i] = current_streak

    return pd.Series(counts, index=data.index)


def calculate_higher_highs(data, min_consecutive=3):
    # Calculate positive price differences
    price_rising = data['close'].diff().gt(0)

    # Create rolling window to check for consecutive rises
    window_size = min_consecutive - 1
    return price_rising.rolling(
        window=window_size,
        min_periods=window_size
    ).apply(lambda x: x.all()).fillna(0).astype(int)


def is_trendline_breakout(data, lookback=20):
    high = data['high'].values
    close = data['close'].values
    n = len(data)
    breakout = np.zeros(n, dtype=int)

    # Precompute pivot highs
    pivot_mask = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        pivot_mask[i] = (high[i] > high[i - 1]) and (high[i] > high[i + 1])

    # Track valid pivots with sliding window
    valid_pivots = []
    for i in range(n):
        # Remove outdated pivots
        while valid_pivots and valid_pivots[0] < (i - lookback):
            valid_pivots.pop(0)

        # Add new pivot if current is pivoted
        if pivot_mask[i]:
            valid_pivots.append(i)

        # Need at least 2 pivots for trendline
        if len(valid_pivots) >= 2:
            # Get two most recent pivots
            p1, p2 = valid_pivots[-2], valid_pivots[-1]

            # Calculate trendline equation
            slope = (high[p2] - high[p1]) / (p2 - p1)
            intercept = high[p2] - slope * p2

            # Projected price at current position
            expected_high = slope * i + intercept

            # Breakout conditions
            if (close[i] > expected_high) and (close[i] > close[i - 1]):
                breakout[i] = 1

    # Handle initial values
    breakout[:lookback] = 0
    return breakout


def get_expected_high(pivot_highs):
    n = len(pivot_highs)
    sum_x = sum(range(n))
    sum_y = sum(pivot_highs)
    sum_xy = sum(i * pivot_highs[i] for i in range(n))
    sum_x2 = sum(i * i for i in range(n))

    # Calculate the slope and intercept of the linear regression
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    intercept = (sum_y - slope * sum_x) / n

    # Project the expected high for the next period (n+1)
    return slope * n + intercept


def calculate_cumulative_spike(data, period=10, threshold=0.55):
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


def calculate_cumulative_change(data, last_change_length=5):
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


def is_parabolic_sar_bullish(data, period, acceleration):
    close = data['close'].values
    n = len(close)
    sar_bullish = np.zeros(n, dtype=int)

    # Pre-allocate window buffer
    window = np.zeros(period)
    prev_sar = np.zeros(n)
    uptrend = np.zeros(n, dtype=bool)
    extreme_point = np.zeros(n)

    # Main processing loop
    for i in range(1, n):
        if i < period:
            # Initialize values for first period
            prev_sar[i] = close[i - 1]
            uptrend[i] = True
            extreme_point[i] = close[i - 1]
            continue

        # Roll window buffer
        window[:-1] = window[1:]
        window[-1] = close[i - 1]

        # Initialize from buffer
        current_sar = prev_sar[i - 1]
        current_uptrend = uptrend[i - 1]
        current_extreme = extreme_point[i - 1]

        # Update SAR for current position
        if current_uptrend:
            current_sar += acceleration * (current_extreme - current_sar)
            if window[-1] < current_sar:
                current_uptrend = False
                current_sar = current_extreme
                current_extreme = window[-1]
            else:
                current_extreme = max(current_extreme, window[-1])
        else:
            current_sar -= acceleration * (current_sar - current_extreme)
            if window[-1] > current_sar:
                current_uptrend = True
                current_sar = current_extreme
                current_extreme = window[-1]
            else:
                current_extreme = min(current_extreme, window[-1])

        # Store values for next iteration
        prev_sar[i] = current_sar
        uptrend[i] = current_uptrend
        extreme_point[i] = current_extreme

        # Check bullish condition
        sar_bullish[i] = int(close[i] > current_sar)

    # Set first period values to 0
    sar_bullish[:period] = 0
    return sar_bullish


def calculate_keltner_breakout(data, ema_period=20, atr_period=20, multiplier=0.2):
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


def calculate_elder_ray_index(data, ema_period=12):
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
        'sma_crossover', 'macd_line', 'trix',
        'rsi', 'roc', 'momentum', 'cmo',
        'bollinger_bands',
        'positive_closes', 'higher_highs', 'trendline_breakout',
        'cumulative_spike', 'cumulative_change',
        'parabolic_sar_bullish', 'keltner_breakout', 'elder_ray_index', 'atr'
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