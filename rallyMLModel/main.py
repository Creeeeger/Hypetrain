import numpy as np
import pandas as pd
import tensorflow as tf
import tf2onnx
from keras import Input, Model
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.src.metrics import Precision, Recall, AUC
from keras.src.regularizers import regularizers
from sklearn.model_selection import train_test_split

gpus = tf.config.list_physical_devices('GPU')  # Get GPU list
if gpus:
    try:
        for gpu in gpus:
            # Enable memory growth to ensure enough memory is available
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# Enable logging of device placement
# tf.debugging.set_log_device_placement(True)

# 1. Feature Engineering
def create_features(df):
    df = df.copy()

    df['returns'] = df['close'].pct_change()

    df['short_sma'] = calculate_sma(df['close'], 9)
    df['long_sma'] = calculate_sma(df['close'], 21)
    df['sma_crossover'] = (
            (df['short_sma'] > df['long_sma']) &
            (df['short_sma'].shift(1) <= df['long_sma'].shift(1))
    ).astype(int)

    df['sma'] = calculate_sma(df['close'], 20)
    df['price_above_sma'] = (
            (df['close'] > df['sma']) &
            (df['close'].shift(1) <= df['sma'].shift(1))
    ).astype(int)

    df['macd_line'] = calculate_ema(df['close'], 6) - calculate_ema(df['close'], 13)
    df['signal_line'] = calculate_ema(df['macd_line'], 5)
    df['macd_histogram'] = df['macd_line'] - df['signal_line']

    df['trix'] = calculate_trix(df, period=5)

    df['rsi'] = calculate_rsi(df, period=15)

    df['roc'] = calculate_roc(df, period=20)

    df['momentum'] = calculate_momentum(df, period=10)

    df['cmo'] = calculate_cmo(df, period=20)

    df['bollinger_bands'] = calculate_bollinger_bands(df, period=20)

    df['positive_closes'] = df.apply(lambda row: consecutive_positive_closes(row, df, dip_tolerance=0.2), axis=1)

    df['higher_highs'] = df.apply(lambda row: is_higher_highs(df[:row.name + 1], min_consecutive=3), axis=1)

    df['trendline_breakout'] = df.apply(lambda row: is_trendline_breakout(row, df, lookback=20), axis=1)

    df['cumulative_spike'] = df.apply(lambda row: is_cumulative_spike(row, df, period=10, threshold=0.55), axis=1)

    df['cumulative_change'] = df.apply(lambda row: cumulative_percentage_change(row, df, last_change_length=5), axis=1)

    df['parabolic_sar_bullish'] = df.apply(lambda row: is_parabolic_sar_bullish(row, df, period=20, acceleration=0.01),
                                           axis=1)

    df['keltner_breakout'] = df.apply(
        lambda row: is_keltner_breakout(row, df, ema_period=20, atr_period=20, multiplier=0.2), axis=1)

    df['elder_ray_index'] = df.apply(lambda row: elder_ray_index(row, df, ema_period=12), axis=1)

    df['atr'] = calculate_atr(df, period=20)

    # label rows which are spikes (0.8% spike)
    df['target'] = (df['close'].shift(-10) / df['close'] - 1 >= 0.008).astype(int)

    df.dropna(inplace=True)

    return df


def calculate_sma(series, period):
    return series.rolling(window=period).mean()


def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calculate_cmo(df, period):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).sum()
    loss = -delta.where(delta < 0, 0).rolling(window=period).sum()
    cmo = (gain - loss) / (gain + loss) * 100
    return cmo


def calculate_momentum(df, period):
    momentum = df['close'].diff(period)
    return momentum


def calculate_roc(df, period):
    return df['close'].pct_change(periods=period) * 100


def calculate_rsi(df, period):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_trix(df, period):
    # Calculate Triple EMA
    ema1 = calculate_ema(df['close'], period)
    ema2 = calculate_ema(ema1, period)
    ema3 = calculate_ema(ema2, period)

    # TRIX: Percentage rate of change
    trix = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    return trix


def calculate_bollinger_bands(df, period):
    if len(df) < period:
        return np.full((len(df), 4), np.nan)

    # Precompute rolling metrics in one pass
    close = df['close'].values
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


def consecutive_positive_closes(row, df, dip_tolerance):
    idx = row.name  # Get current row index

    if idx == 0:
        return 0  # First row cannot have previous consecutive closes

    count = 0
    for i in range(idx, -1, -1):  # Iterate backward from current row
        if df['returns'].iloc[i] * 100 > 0:
            count += 1  # Increase count for consecutive positive close
        elif df['returns'].iloc[i] * 100 < -dip_tolerance:
            break  # Reset if there's a dip beyond tolerance

    return count


def is_higher_highs(df, min_consecutive):
    if len(df) < min_consecutive:
        return 0  # Not enough data to evaluate

    for i in range(len(df) - min_consecutive, len(df) - 1):
        if df['close'].iloc[i + 1] <= df['close'].iloc[i]:
            return 0

    return 1


def is_trendline_breakout(row, df, lookback):
    # Get the index of the current row
    idx = row.name

    # Ensure we have enough data for the lookback period
    if idx < lookback:
        return 0

    # Find pivot highs for the trend line
    pivot_highs = []
    for i in range(lookback):
        if idx - i - 1 >= 0 and idx - i + 1 < len(df):  # Ensure indices are within bounds
            p = df.iloc[idx - i]
            if p['high'] > df.iloc[idx - i - 1]['high'] and p['high'] > df.iloc[idx - i + 1]['high']:
                pivot_highs.append(p['high'])

    if len(pivot_highs) < 2:
        return 0

    # Calculate the expected high using linear regression
    expected_high = get_expected_high(pivot_highs)
    current_close = row['close']

    # Return 1 if breakout condition is met
    return 1 if current_close > expected_high and current_close > df['close'].iloc[idx - 1] else 0


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


def is_cumulative_spike(row, df, period, threshold):
    # Get the index of the current row
    idx = row.name

    # Ensure we have enough data for the lookback period
    if idx < period:
        return 0  # Not enough data

    # Compute cumulative return using compounding formula
    cumulative_return = ((df['returns'].iloc[idx - period:idx] + 1).prod() - 1) * 100

    # Return 1 if it meets the threshold, otherwise 0
    return 1 if cumulative_return >= threshold else 0


def cumulative_percentage_change(row, df, last_change_length):
    # Get the index of the current row
    idx = row.name

    # Ensure there are enough previous rows to compute change
    if idx < last_change_length:
        return 0  # Not enough data

    # Compute cumulative percentage change using compounding formula
    return ((df['returns'].iloc[idx - last_change_length:idx] + 1).prod() - 1) * 100


def is_parabolic_sar_bullish(row, df, period, acceleration):
    idx = row.name  # Get the index of the current row

    # Ensure we have enough data for the period
    if idx < period:
        return 0  # Not enough data to compute SAR

    # Use the last 'period' number of data points (rows) up to the current row
    df_period = df.iloc[idx - period:idx]

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


def is_keltner_breakout(row, df, ema_period, atr_period, multiplier):
    ema = df['close'].ewm(span=ema_period, adjust=False).mean()
    atr = calculate_atr(df, atr_period)

    # Compute upper band
    upper_band = ema + (multiplier * atr)

    # Check if the close price of the given row is greater than the upper band
    return 1 if row['close'] > upper_band.loc[row.name] else 0


def elder_ray_index(row, df, ema_period):
    ema = df['close'].ewm(span=ema_period, adjust=False).mean()
    return row['close'] - ema.loc[row.name]


def calculate_atr(df, period):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())

    true_range = np.maximum(high_low, high_close, low_close)
    atr = true_range.rolling(window=period).mean()
    return atr


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer


def prepare_sequences(data, features, window_size):
    # Identify features that need special handling
    numeric_features = data[features].select_dtypes(include=['float64', 'int64']).columns
    low_range_features = []
    other_features = []

    # Detect features with small value ranges (adjust threshold as needed)
    for feature in numeric_features:
        value_range = data[feature].max() - data[feature].min()
        if value_range < 1:  # Adjust this threshold based on your data
            low_range_features.append(feature)
        else:
            other_features.append(feature)

    # Create different scaling pipelines for different feature types
    preprocessor = ColumnTransformer(
        transformers=[
            ('low_range', StandardScaler(), low_range_features),
            ('default', MinMaxScaler(), other_features)
        ])

    scaled_data = preprocessor.fit_transform(data[features])

    # Convert scaled data into DataFrame
    scaled_df = pd.DataFrame(scaled_data, columns=low_range_features + other_features)

    # print(scaled_df.head(200).to_string())

    # Sequence creation remains the same
    x, y = [], []
    for i in range(window_size, len(data)):
        x.append(scaled_data[i - window_size:i])
        y.append(data['target'].iloc[i])

    return np.array(x), np.array(y), preprocessor

def build_spike_model(input_shape):
    l2_regularizer = regularizers.L2(0.01)

    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=False, kernel_regularizer=l2_regularizer, recurrent_activation="tanh")(inputs)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
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

def train_spike_predictor(data_path):
    # read the data of the CSV file and skip bad lines to prevent errors
    data_of_csv = pd.read_csv(data_path, on_bad_lines='skip')

    # Engineer the features
    data_of_csv = create_features(data_of_csv)

    # set options so that the whole table can be printed out
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_rows', None)

    # debug print engineered features
    print(data_of_csv)

    # feature categories from Java feature creator
    features = [
        'sma_crossover', 'price_above_sma', 'macd_line', 'trix',
        'rsi', 'roc', 'momentum', 'cmo',
        'bollinger_bands',
        'positive_closes', 'higher_highs', 'trendline_breakout',
        'cumulative_spike', 'cumulative_change',
        'parabolic_sar_bullish', 'keltner_breakout', 'elder_ray_index', 'atr'
    ]

    x, y, scaler = prepare_sequences(data_of_csv, features, len(features))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = build_spike_model((x_train.shape[1], x_train.shape[2]))

    early_stop = EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

    with tf.device('/CPU:0'):
        model.fit(
            x_train, y_train,
            epochs=1,
            batch_size=64,
            validation_data=(x_test, y_test),
            callbacks=[early_stop, reduce_lr]
        )

    # Define the input signature dynamically based on training data shape
    input_signature = [tf.TensorSpec([None, *x_train.shape[1:]], tf.float32)]

    # Convert and save the ONNX model directly
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature)
    with open("spike_predictor.onnx", "wb") as f:
        f.write(model_proto.SerializeToString())

    return model, scaler


if __name__ == "__main__":
    model, scaler = train_spike_predictor('highFrequencyStocks.csv')  # Main function for training
    print("Training done!")

    # 1. add sma remember
    # 2. fix scaler
    # 3. rework network architecture
    # 4. rework training process
    # 5. improve training speed