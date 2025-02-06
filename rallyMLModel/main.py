import os

import numpy as np
import tensorflow as tf
import tf2onnx
from keras import Input, Model
from keras import metrics
from keras.src.callbacks.early_stopping import EarlyStopping
from keras.src.layers.core.dense import Dense
from keras.src.layers.rnn.lstm import LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# 1. Feature Engineering
def create_features(df):
    df = df.copy()

    df['returns'] = df['close'].pct_change()

    df['short_sma'] = calculate_sma(df['close'], 15)
    df['long_sma'] = calculate_sma(df['close'], 30)
    df['sma_crossover'] = (
            (df['short_sma'] > df['long_sma']) &
            (df['short_sma'].shift(1) <= df['long_sma'].shift(1))
    ).astype(int)

    df['short_ema'] = calculate_ema(df['close'], 15)
    df['long_ema'] = calculate_ema(df['close'], 30)
    df['ema_crossover'] = (
            (df['short_ema'] > df['long_ema']) &
            (df['short_ema'].shift(1) <= df['long_ema'].shift(1))
    ).astype(int)

    df['sma'] = calculate_sma(df['close'], 30)
    df['price_above_sma'] = (
            (df['close'] > df['sma']) &
            (df['close'].shift(1) <= df['sma'].shift(1))
    ).astype(int)

    df['macd_line'] = calculate_ema(df['close'], 12) - calculate_ema(df['close'], 30)
    df['signal_line'] = calculate_ema(df['macd_line'], 9)
    df['macd_histogram'] = df['macd_line'] - df['signal_line']

    df['trix'] = calculate_trix(df, period=30)

    df['kama'] = calculate_kama(df, period=30)

    df['rsi'] = calculate_rsi(df, period=30)

    df['roc'] = calculate_roc(df, period=30)

    df['momentum'] = calculate_momentum(df, period=30)

    df['cmo'] = calculate_cmo(df, period=30)

    df['acceleration'] = df.apply(lambda row: calculate_acceleration(df, period=30), axis=1)

    df['bollinger_bands'] = calculate_bollinger_bands(df, period=30)

    df['breakout'] = df.apply(lambda row: is_breakout(df, resistance_period=30), axis=1)

    df['donchian_breakout'] = df.apply(lambda row: donchian_breakout(df, period=30), axis=1)

    df['volatility_spike'] = df.apply(lambda row: is_volatility_spike(df, period=30), axis=1)

    df['volatility_ratio'] = df.apply(lambda row: rolling_volatility_ratio(df, short_period=3, long_period=5), axis=1)

    df['positive_closes'] = df.apply(lambda row: consecutive_positive_closes(df, dip_tolerance=0.5), axis=1)

    df['higher_highs'] = df.apply(lambda row: is_higher_highs(df, min_consecutive=3), axis=1)

    df['fractal_breakout'] = df.apply(
        lambda row: is_fractal_breakout(df, consolidation_period=3, volatility_threshold=0.05), axis=1)

    df['candle_patterns'] = df.apply(lambda row: detect_candle_patterns(df.iloc[-1], df.iloc[-2]), axis=1)

    df['trendline_breakout'] = df.apply(lambda row: is_trendline_breakout(df, lookback=4), axis=1)

    df['zscore_spike'] = df.apply(lambda row: is_zscore_spike(df, period=30), axis=1)

    df['cumulative_spike'] = df.apply(lambda row: is_cumulative_spike(df, period=30, threshold=10.0), axis=1)

    df['cumulative_change'] = df.apply(lambda row: cumulative_percentage_change(df, last_change_length=30), axis=1)

    df['breakout_above_ma'] = df.apply(lambda row: is_breakout_above_ma(df, period=30, use_ema=False), axis=1)

    df['parabolic_sar_bullish'] = df.apply(lambda row: is_parabolic_sar_bullish(), axis=1)

    df['keltner_breakout'] = df.apply(lambda row: is_keltner_breakout(df, ema_period=30, atr_period=30), axis=1)

    df['elder_ray_index'] = df.apply(lambda row: elder_ray_index(df, ema_period=30), axis=1)

    df['volume_spike'] = df.apply(lambda row: is_volume_spike(df, period=30, threshold_factor=2.0), axis=1)

    df['atr'] = df.apply(lambda row: calculate_atr(df, period=30), axis=1)

    df['target'] = (df['close'].shift(-15) / df['close'] - 1 >= 0.03).astype(int)

    df.dropna(axis=0, how='all', inplace=True)

    return df


# 1. Simple Moving Average (SMA) Crossovers
def calculate_sma(series, period):
    return series.rolling(window=period).mean()


# 2. EMA
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
    roc = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
    return roc


def calculate_rsi(df, period):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_efficiency_ratio(prices, period):
    direction = abs(prices[-1] - prices[-period])
    volatility = sum(abs(prices[i] - prices[i - 1]) for i in range(1, period))
    return direction / volatility if volatility != 0 else 0


def calculate_kama(df, period):
    prices = df['close'].tolist()
    efficiency_ratio = calculate_efficiency_ratio(prices, period)
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    smooth_sc = efficiency_ratio * (fast_sc - slow_sc) + slow_sc

    kama = [np.mean(prices[:period])]
    for i in range(period, len(prices)):
        kama.append(kama[-1] + smooth_sc * (prices[i] - kama[-1]))

    return pd.Series(kama, index=df.index[period - 1:])


def calculate_trix(df, period):
    # Calculate Triple EMA
    ema1 = calculate_ema(df['close'], period)
    ema2 = calculate_ema(ema1, period)
    ema3 = calculate_ema(ema2, period)

    # TRIX: Percentage rate of change
    trix = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    return trix


def calculate_acceleration(df, period):
    # Edge case: If the DataFrame is too small, return NaN
    if len(df) < period + 2:
        return np.nan

    # Calculate momentum for the entire DataFrame
    momentum = df['close'].diff(period)  # Equivalent to calculate_momentum

    # Ensure there are enough values to compute acceleration
    if len(momentum) < 3:
        return np.nan

    # Calculate acceleration using the second derivative (central difference)
    acceleration = (momentum.iloc[-1] - 2 * momentum.iloc[-2] + momentum.iloc[-3]) / (period ** 2)
    return acceleration


import pandas as pd


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

    # Vectorized calculations
    upper = rolling_mean + 2 * rolling_std
    lower = rolling_mean - 2 * rolling_std
    bandwidth = (upper - lower) / np.where(rolling_mean != 0, rolling_mean, np.nan)

    return bandwidth


def is_breakout(df, resistance_period):
    if len(df) < resistance_period + 1:
        return 0

    current_close = df['close'].iloc[-1]
    resistance = df['close'].iloc[-resistance_period:].max()

    # Check if current close breaks resistance
    if current_close > resistance >= df['close'].iloc[-2]:
        return 1
    return 0


def donchian_breakout(df, period):
    if len(df) < period:
        return 0

    # Get the current close price
    current_close = df['close'].iloc[-1]

    # Get the maximum close price in the past 'period' values
    current_max = df['close'].iloc[-period:].max()

    # Check for breakout (current close exceeds the maximum of the previous 'period' values)
    if current_close > current_max >= df['close'].iloc[-2]:
        return 1
    return 0


def is_volatility_spike(df, period):
    if len(df) < period:
        return 0

    current_change = df['returns'].iloc[-1]
    std_dev = df['returns'].iloc[-period:].std()

    # Check if current change is more than 2 standard deviations
    if abs(current_change) > 2 * std_dev:
        return 1
    return 0


def rolling_volatility_ratio(df, short_period, long_period):
    if len(df) < max(short_period, long_period) * 2:
        return 0

    short_term = np.mean(df['bollinger_bands'].iloc[-short_period:])
    long_term = np.mean(df['bollinger_bands'].iloc[-long_period:])

    return short_term / long_term if long_term > 0 else 0


# 17. Consecutive Positive Closes with Momentum Tolerance
def consecutive_positive_closes(df, dip_tolerance):
    # Calculate percentage changes between consecutive closes
    changes = (df['close'].pct_change() * 100)

    # Use a boolean mask to identify positive changes
    positive_changes = changes > 0

    # Use the cumulative sum to find consecutive positive changes
    consecutive_count = positive_changes.cumsum()

    # Reset count to 0 when the change is less than the dip tolerance
    consecutive_count[changes < -dip_tolerance] = 0

    # Return the maximum consecutive count
    return consecutive_count.max()


# 18. Higher Highs Pattern with Adaptive Window
def is_higher_highs(df, min_consecutive):
    if len(df) < min_consecutive:
        return 0

    for i in range(len(df) - min_consecutive, len(df) - 1):
        if df['close'].iloc[i + 1] <= df['close'].iloc[i]:
            return 0

    return 1


# 19. Fractal Breakout Detection with Consolidation Phase
def is_fractal_breakout(df, consolidation_period, volatility_threshold):
    if len(df) < consolidation_period + 2:
        return 0

    # Calculate consolidation range
    consolidation_high = df['high'].iloc[-consolidation_period:].max()
    consolidation_low = df['low'].iloc[-consolidation_period:].min()
    current_close = df['close'].iloc[-1]
    range_size = consolidation_high - consolidation_low

    # Return 1 if breakout condition is met
    return 1 if current_close > consolidation_high and range_size / consolidation_low < volatility_threshold else 0


def detect_candle_patterns(current, previous):
    pattern_mask = 0

    # Hammer detection
    is_hammer = (current['high'] - current['low']) > 3 * (current['close'] - current['open']) and \
                (current['close'] > current['open']) and \
                (current['close'] - current['low']) > 0.7 * (current['high'] - current['low'])

    # Bullish Engulfing
    is_engulfing = (previous['close'] < previous['open']) and \
                   (current['close'] > previous['open']) and \
                   (current['open'] < previous['close'])

    # Morning Star (simplified)
    is_morning_star = (previous['close'] < previous['open']) and \
                      (current['open'] > previous['close']) and \
                      (current['close'] > previous['open'])

    if is_hammer:
        pattern_mask |= 0b1
    if is_engulfing:
        pattern_mask |= 0b10
    if is_morning_star:
        pattern_mask |= 0b100

    return pattern_mask


# 21. Automated Trend-line Analysis
def is_trendline_breakout(df, lookback):
    if len(df) < lookback + 2:
        return 0

    # Find pivot highs for trend line
    pivot_highs = []
    for i in range(3, lookback):
        p = df.iloc[-i]
        if p['high'] > df.iloc[-i - 1]['high'] and p['high'] > df.iloc[-i + 1]['high']:
            pivot_highs.append(p['high'])

    if len(pivot_highs) < 2:
        return 0

    # Linear regression of pivot highs
    expected_high = get_expected_high(pivot_highs)
    current_close = df['close'].iloc[-1]

    # Return 1 if breakout condition is met
    return 1 if current_close > expected_high and current_close > df['close'].iloc[-2] else 0


def get_expected_high(pivot_highs):
    n = len(pivot_highs)
    sum_x = sum(range(n))
    sum_y = sum(pivot_highs)
    sum_xy = sum(i * pivot_highs[i] for i in range(n))
    sum_x2 = sum(i * i for i in range(n))

    # Calculate the slope and intercept of the linear regression
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    intercept = (sum_y - slope * sum_x) / n

    # Project the expected high
    return slope * (n + 1) + intercept


# 22. Z-Score of Returns using Incremental Calculation
def is_zscore_spike(df, period):
    # Ensure we have enough data points
    if len(df) < period:
        return 0

    # Calculate the rolling mean and standard deviation for the last 'period' data points
    recent_returns = df['returns'].iloc[-period:]
    mean = np.mean(recent_returns)
    std_dev = np.std(recent_returns)

    # Calculate the Z-score of the most recent return
    current_return = df['returns'].iloc[-1]

    # Return 1 if Z-score is greater than or equal to 2.0, else 0
    return 1 if std_dev != 0 and (current_return - mean) / std_dev >= 2.0 else 0


# 23. Cumulative Percentage Change with Threshold Check
def is_cumulative_spike(df, period, threshold):
    if len(df) < period:
        return 0

    # Sum the percentage changes over the period
    cumulative_change = df['returns'].iloc[-period:].sum()

    # Return 1 if true, 0 if false
    return 1 if cumulative_change >= threshold else 0


# 24. Cumulative Percentage Change (simple sum)
def cumulative_percentage_change(df, last_change_length):
    start_index = len(df) - last_change_length
    return df['returns'].iloc[start_index:].sum()


# 25. Breakout Above Moving Average
def is_breakout_above_ma(df, period, use_ema=False):
    if len(df) < period:
        return 0

    # Get current and previous close prices
    current_close = df['close'].iloc[-1]
    previous_close = df['close'].iloc[-2] if len(df) >= 2 else current_close

    # Calculate the moving average (SMA or EMA)
    current_ma = calculate_ma(df, period, use_ema)
    previous_ma = calculate_ma(df[:-1], period, use_ema)

    # Return 1 if breakout occurs, 0 otherwise
    return 1 if current_close > current_ma and previous_close <= previous_ma else 0


def is_parabolic_sar_bullish():
    return 1  # needed to use


def is_keltner_breakout(df, ema_period, atr_period, multiplier=1.5):
    ema = df['close'].ewm(span=ema_period, adjust=False).mean().iloc[-1]
    atr = calculate_atr(df, atr_period)
    upper_band = ema + (multiplier * atr)

    return 1 if df['close'].iloc[-1] > upper_band else 0


def elder_ray_index(df, ema_period):
    ema = df['close'].ewm(span=ema_period, adjust=False).mean().iloc[-1]
    return df['close'].iloc[-1] - ema


def is_volume_spike(df, period, threshold_factor=2.0):
    if len(df) < period:
        return 0

    avg_volume = df['volume'].iloc[-period:].mean()
    current_volume = df['volume'].iloc[-1]

    return 1 if current_volume > (avg_volume * threshold_factor) else 0


def calculate_atr(df, period):
    # Calculate the True Range using vectorized operations
    high = df['high']
    low = df['low']
    prev_close = df['close'].shift(1)

    true_range = np.maximum(high - low,
                            np.abs(high - prev_close),
                            np.abs(low - prev_close))

    # Calculate the ATR using a rolling window
    atr = true_range.rolling(window=period).mean()

    return atr.iloc[-1]  # Return the latest ATR value


def calculate_ma(df, period, use_ema=False):
    if use_ema:
        return df['close'].ewm(span=period, adjust=False).mean().iloc[-1]
    else:
        return df['close'].iloc[-period:].mean()

# 2. Data Preparation
def prepare_sequences(data, features, target, window_size=30):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[features])

    x, y = [], []
    for i in range(window_size, len(data)):
        x.append(scaled_data[i - window_size:i])
        y.append(data[target].iloc[i])

    return np.array(x), np.array(y), scaler


# 3. Model Architecture
def build_spike_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(64, unroll=True)(inputs)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid', dtype='float32')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy',
                           metrics.Precision(name='precision'),
                           metrics.Recall(name='recall')])
    return model

# 4. Training Pipeline
def train_spike_predictor(data_path):
    # Load and preprocess data
    df = pd.read_csv(data_path, on_bad_lines='skip')

    df = create_features(df)

    # Feature selection
    features = [
        'sma_crossover', 'ema_crossover', 'price_above_sma', 'macd_line', 'trix', 'kama',
        'rsi', 'roc', 'momentum', 'cmo', 'acceleration',
        'bollinger_bands', 'breakout', 'donchian_breakout', 'volatility_spike', 'volatility_ratio', 'positive_closes',
        'higher_highs', 'fractal_breakout', 'candle_patterns', 'trendline_breakout', 'zscore_spike', 'cumulative_spike',
        'cumulative_change', 'breakout_above_ma', 'parabolic_sar_bullish', 'keltner_breakout', 'elder_ray_index',
        'volume_spike', 'atr'
    ]

    target = 'target'

    # Create sequences
    x, y, scaler = prepare_sequences(df, features, target)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Build and train model
    model = build_spike_model((x_train.shape[1], x_train.shape[2]))

    early_stop = EarlyStopping(monitor='val_precision', patience=5, mode='max', restore_best_weights=True)

    model.fit(
        x_train, y_train,
        epochs=1,
        batch_size=128,
        validation_data=(x_test, y_test),
        callbacks=[early_stop]
    )

    # Save for production
    model.save('spike_predictor.keras')

    # Ensure the model is built before exporting
    input_signature = [tf.TensorSpec([None, *(x_train.shape[1], x_train.shape[2])], tf.float32)]

    # Convert and save the ONNX model
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature)
    with open("spike_predictor.onnx", "wb") as f:
        f.write(model_proto.SerializeToString())

    os.remove("spike_predictor.keras")

    return model, scaler

if __name__ == "__main__":
    model, scaler = train_spike_predictor('high_frequency_stocks.csv')
    print("Training done!")