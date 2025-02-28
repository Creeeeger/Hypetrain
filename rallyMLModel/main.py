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
from sklearn.utils import compute_class_weight


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

    data_of_csv['positive_closes'] = data_of_csv.apply(
        lambda row: consecutive_positive_closes(row, data_of_csv, dip_tolerance=0.2), axis=1)

    data_of_csv['higher_highs'] = data_of_csv.apply(
        lambda row: is_higher_highs(data_of_csv[:row.name + 1], min_consecutive=3), axis=1)

    data_of_csv['trendline_breakout'] = data_of_csv.apply(
        lambda row: is_trendline_breakout(row, data_of_csv, lookback=20), axis=1)

    data_of_csv['cumulative_spike'] = data_of_csv.apply(
        lambda row: is_cumulative_spike(row, data_of_csv, period=10, threshold=0.55), axis=1)

    data_of_csv['cumulative_change'] = data_of_csv.apply(
        lambda row: cumulative_percentage_change(row, data_of_csv, last_change_length=5), axis=1)

    data_of_csv['parabolic_sar_bullish'] = data_of_csv.apply(
        lambda row: is_parabolic_sar_bullish(row, data_of_csv, period=20, acceleration=0.01),
        axis=1)

    data_of_csv['keltner_breakout'] = data_of_csv.apply(
        lambda row: is_keltner_breakout(row, data_of_csv, ema_period=20, atr_period=20, multiplier=0.2), axis=1)

    data_of_csv['elder_ray_index'] = data_of_csv.apply(lambda row: elder_ray_index(row, data_of_csv, ema_period=12),
                                                       axis=1)

    data_of_csv['atr'] = calculate_atr(data_of_csv, period=20)

    # label rows which are spikes (0.8% spike)
    data_of_csv['target'] = (data_of_csv['close'].shift(-10) / data_of_csv['close'] - 1 >= 0.008).astype(int)

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


def consecutive_positive_closes(row, data_of_csv, dip_tolerance):
    idx = row.name  # Get current row index

    if idx == 0:
        return 0  # First row cannot have previous consecutive closes

    count = 0
    for i in range(idx, -1, -1):  # Iterate backward from current row
        if data_of_csv['returns'].iloc[i] * 100 > 0:
            count += 1  # Increase count for consecutive positive close
        elif data_of_csv['returns'].iloc[i] * 100 < -dip_tolerance:
            break  # Reset if there's a dip beyond tolerance

    return count


def is_higher_highs(data_of_csv, min_consecutive):
    if len(data_of_csv) < min_consecutive:
        return 0  # Not enough data to evaluate

    for i in range(len(data_of_csv) - min_consecutive, len(data_of_csv) - 1):
        if data_of_csv['close'].iloc[i + 1] <= data_of_csv['close'].iloc[i]:
            return 0

    return 1


def is_trendline_breakout(row, data_of_csv, lookback):
    # Get the index of the current row
    idx = row.name

    # Ensure we have enough data for the lookback period
    if idx < lookback:
        return 0

    # Find pivot highs for the trend line
    pivot_highs = []
    for i in range(lookback):
        if idx - i - 1 >= 0 and idx - i + 1 < len(data_of_csv):  # Ensure indices are within bounds
            p = data_of_csv.iloc[idx - i]
            if p['high'] > data_of_csv.iloc[idx - i - 1]['high'] and p['high'] > data_of_csv.iloc[idx - i + 1]['high']:
                pivot_highs.append(p['high'])

    if len(pivot_highs) < 2:
        return 0

    # Calculate the expected high using linear regression
    expected_high = get_expected_high(pivot_highs)
    current_close = row['close']

    # Return 1 if breakout condition is met
    return 1 if current_close > expected_high and current_close > data_of_csv['close'].iloc[idx - 1] else 0


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


def is_cumulative_spike(row, data_of_csv, period, threshold):
    # Get the index of the current row
    idx = row.name

    # Ensure we have enough data for the lookback period
    if idx < period:
        return 0  # Not enough data

    # Compute cumulative return using compounding formula
    cumulative_return = ((data_of_csv['returns'].iloc[idx - period:idx] + 1).prod() - 1) * 100

    # Return 1 if it meets the threshold, otherwise 0
    return 1 if cumulative_return >= threshold else 0


def cumulative_percentage_change(row, data_of_csv, last_change_length):
    # Get the index of the current row
    idx = row.name

    # Ensure there are enough previous rows to compute change
    if idx < last_change_length:
        return 0  # Not enough data

    # Compute cumulative percentage change using compounding formula
    return ((data_of_csv['returns'].iloc[idx - last_change_length:idx] + 1).prod() - 1) * 100


def is_parabolic_sar_bullish(row, data_of_csv, period, acceleration):
    idx = row.name  # Get the index of the current row

    # Ensure we have enough data for the period
    if idx < period:
        return 0  # Not enough data to compute SAR

    # Use the last 'period' number of data points (rows) up to the current row
    df_period = data_of_csv.iloc[idx - period:idx]

    # Initial values
    prev_sar = df_period.iloc[0]['close']  # First SAR value from the period
    uptrend = True
    extreme_point = df_period.iloc[0]['close']  # Start from the first close price

    # Loop through only the 'period' length data
    for i in range(1, period):
        current_close = df_period.iloc[i]['close']

        if uptrend:
            current_sar = prev_sar + acceleration * (extreme_point - prev_sar)
            if current_close < current_sar:
                # Switch to downtrend
                uptrend = False
                prev_sar = extreme_point  # Reset SAR
                extreme_point = current_close
            else:
                prev_sar = current_sar  # Update SAR for uptrend
                extreme_point = max(extreme_point, current_close)
        else:
            current_sar = prev_sar - acceleration * (prev_sar - extreme_point)
            if current_close > current_sar:
                # Switch to uptrend
                uptrend = True
                prev_sar = extreme_point  # Reset SAR
                extreme_point = current_close
            else:
                prev_sar = current_sar  # Update SAR for downtrend
                extreme_point = min(extreme_point, current_close)

    # Compare the last closing price with the final SAR value
    return 1 if row['close'] > prev_sar else 0


def is_keltner_breakout(row, data_of_csv, ema_period, atr_period, multiplier):
    ema = data_of_csv['close'].ewm(span=ema_period, adjust=False).mean()
    atr = calculate_atr(data_of_csv, atr_period)

    # Compute upper band
    upper_band = ema + (multiplier * atr)

    # Check if the close price of the given row is greater than the upper band
    return 1 if row['close'] > upper_band.loc[row.name] else 0


def elder_ray_index(row, data_of_csv, ema_period):
    ema = data_of_csv['close'].ewm(span=ema_period, adjust=False).mean()
    return row['close'] - ema.loc[row.name]


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

    feature, label = [], []
    for i in range(window_size, len(data)):
        feature.append(scaled_data[i - window_size:i])
        label.append(data['target'].iloc[i])

    return np.array(feature), np.array(label), preprocessor


# 3. Model architecture construction
def build_model(input_shape):
    l2_regularizer = regularizers.L2(0.001)
    inputs = Input(shape=input_shape)

    # Convolutional block: extract local features from the sequence
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=l2_regularizer)(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
    x = Dropout(0.3)(x)

    # Optional second convolutional block for deeper feature extraction
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=l2_regularizer)(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
    x = Dropout(0.3)(x)

    # LSTM layer to capture the sequential dependencies after CNN processing
    x = LSTM(64, return_sequences=False, kernel_regularizer=l2_regularizer, recurrent_activation="tanh")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Dense layers for further processing before final output
    x = Dense(64, activation='relu', kernel_regularizer=l2_regularizer)(x)
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
    class_weights = compute_class_weight('balanced', classes=np.unique(labels_train_res), y=labels_train_res)
    class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

    print("Class distribution:", np.bincount(labels_train.flatten()))
    print("Unique label values:", np.unique(labels_train))

    # training process of the model CPU has in my case better performance than GPU
    # Train with resampled data
    with tf.device('/CPU:0'):
        model.fit(
            features_train_res, labels_train_res,  # Use resampled data
            epochs=256,
            batch_size=128,
            validation_data=(features_test, labels_test),
            callbacks=[early_stop, reduce_lr],
            class_weight=class_weights_dict  # Optional as classes are balanced
        )

    # Convert and save the ONNX model directly & define the input signature dynamically based on training data shape
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=[
        tf.TensorSpec([None, *features_train.shape[1:]], tf.float32)])
    with open("spike_predictor.onnx", "wb") as f:
        f.write(model_proto.SerializeToString())


if __name__ == "__main__":
    train_spike_predictor('highFrequencyStocks.csv')  # Main function for training
    print("Training done!")