import os  # For environment variables and OS-level settings

import numpy as np  # Numerical computations library
import pandas as pd  # Data manipulation and analysis
import tensorflow as tf  # Machine learning framework
import tf2onnx  # Converts TensorFlow models to ONNX format for interoperability
from imblearn.over_sampling import SMOTE  # For synthetic oversampling to handle class imbalance
from keras import Input, Model  # Keras functional API for defining models
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau  # Training control callbacks
from keras.src.layers import LSTM, Dense, Dropout, BatchNormalization, MaxPooling1D, Conv1D  # Neural net layers
from keras.src.metrics import Precision, Recall, AUC  # Metrics to evaluate model during training
from keras.src.regularizers import regularizers  # Regularizers to reduce overfitting
from sklearn.compose import ColumnTransformer  # Pipeline helper for data preprocessing
from sklearn.preprocessing import MinMaxScaler  # Scale features to 0-1 range


# Install dependencies
# pip install numpy pandas tensorflow tf2onnx imbalanced-learn scikit-learn
# pip install --upgrade scikit-learn imbalanced-learn

# 1. Feature Engineering
def create_features(data_of_csv):
    # Work on a copy to avoid modifying original dataframe unexpectedly
    data_of_csv = data_of_csv.copy()

    # Calculate simple returns: percent change of 'close' price between consecutive rows
    data_of_csv['returns'] = data_of_csv['close'].pct_change(fill_method=None)

    # Calculate short and long Simple Moving Averages (SMA) to identify trends
    data_of_csv['short_sma'] = calculate_sma(data_of_csv['close'], 9)  # 9-day SMA
    data_of_csv['long_sma'] = calculate_sma(data_of_csv['close'], 21)  # 21-day SMA

    # Detect bullish crossover where short SMA crosses above long SMA on current day
    bullish_crossover = (
            (data_of_csv['short_sma'] > data_of_csv['long_sma']) &
            (data_of_csv['short_sma'].shift(1) <= data_of_csv['long_sma'].shift(1))
    )

    # Detect bearish crossover where short SMA crosses below long SMA on current day
    bearish_crossover = (
            (data_of_csv['short_sma'] < data_of_csv['long_sma']) &
            (data_of_csv['short_sma'].shift(1) >= data_of_csv['long_sma'].shift(1))
    )

    # Initialize the column with 0 (no crossover)
    data_of_csv['sma_crossover'] = 0

    # Mark bullish crossover days with +1
    data_of_csv.loc[bullish_crossover, 'sma_crossover'] = 1
    # Mark bearish crossover days with -1
    data_of_csv.loc[bearish_crossover, 'sma_crossover'] = -1

    # Forward fill to propagate last crossover state forward
    # This means in-between crossover days inherit last crossover signal
    data_of_csv['sma_crossover'] = (
        data_of_csv['sma_crossover']
        .replace(0, np.nan)  # Temporarily treat 0 as missing to forward fill properly
        .ffill()  # Forward fill missing values
        .fillna(0)  # Fill initial missing with 0 (neutral)
        .astype(int)  # Convert back to integers
    )

    # Calculate other technical indicators to use as features
    data_of_csv['trix'] = calculate_trix(data_of_csv, period=5)  # Triple exponential moving average rate of change
    data_of_csv['roc'] = calculate_roc(data_of_csv, period=20)  # Rate of change over 20 days
    data_of_csv['bollinger_bands'] = calculate_bollinger_bands(data_of_csv, period=20)  # Bandwidth from Bollinger Bands
    data_of_csv['cumulative_spike'] = calculate_cumulative_spike(data_of_csv, period=10, threshold=0.35)  # Spike signal
    data_of_csv['cumulative_change'] = calculate_cumulative_change(data_of_csv,
                                                                   last_change_length=8)  # Recent cumulative returns
    data_of_csv['keltner_breakout'] = calculate_keltner_breakout(data_of_csv, 12, 10, 0.3,
                                                                 0.4)  # Breakout signal using Keltner channels
    data_of_csv['elder_ray_index'] = calculate_elder_ray_index(data_of_csv, ema_period=12)  # Momentum indicator

    # Ensure target column is integer type (0 or 1 for classification)
    data_of_csv['target'] = data_of_csv['target'].astype(int)

    # Drop rows with missing values after feature calculations to keep dataset clean
    data_of_csv.dropna(inplace=True)

    # Return processed dataframe ready for model training
    return data_of_csv


def calculate_sma(series, period):
    # Calculate simple moving average for a series over specified window length
    return series.rolling(window=period).mean()


def calculate_ema(series, period):
    # Calculate exponential moving average, giving more weight to recent data points
    return series.ewm(span=period, adjust=False).mean()


def calculate_roc(data_of_csv, period):
    # Calculate rate of change as percentage over 'period' days
    return data_of_csv['close'].pct_change(periods=period, fill_method=None) * 100


def calculate_trix(data_of_csv, period):
    # Calculate the Triple Exponential Moving Average (TRIX) indicator
    # TRIX is a momentum oscillator that smooths price changes by applying EMA three times

    # Step 1: Calculate the first EMA of the closing prices with the specified period
    ema1 = calculate_ema(data_of_csv['close'], period)

    # Step 2: Calculate the second EMA by applying EMA on the first EMA
    ema2 = calculate_ema(ema1, period)

    # Step 3: Calculate the third EMA by applying EMA on the second EMA
    ema3 = calculate_ema(ema2, period)

    # Step 4: Calculate the TRIX value as the percentage rate of change of the triple EMA
    # This is done by comparing the current EMA value to the previous day's EMA value
    trix = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100

    # Return the TRIX series (percent change values)
    return trix


def calculate_bollinger_bands(data_of_csv, period):
    # Calculate the Bollinger Bandwidth, which measures the width of Bollinger Bands relative to the moving average

    # If there is not enough data to compute for the specified period, return an array of NaNs
    if len(data_of_csv) < period:
        return np.full((len(data_of_csv), 4), np.nan)

    # Extract closing prices as a numpy array for fast processing
    close = data_of_csv['close'].values

    # Prepare empty arrays to store rolling means and standard deviations
    rolling_mean = np.empty(len(close))
    rolling_std = np.empty(len(close))

    # For each point in the data, calculate the rolling mean and standard deviation over the specified period
    for i in range(len(close)):
        # Define window start index ensuring it doesn't go below zero
        start = max(0, i - period + 1)
        # Extract the window slice from start to current index (inclusive)
        window = close[start:i + 1]

        # Calculate the mean of the values in the window
        rolling_mean[i] = window.mean()
        # Calculate the standard deviation with population formula (ddof=0)
        # If only one value exists, std is 0 (no variance)
        rolling_std[i] = window.std(ddof=0) if len(window) > 1 else 0

    # Compute the upper Bollinger Band: mean + 2 standard deviations
    upper = rolling_mean + 2 * rolling_std
    # Compute the lower Bollinger Band: mean - 2 standard deviations
    lower = rolling_mean - 2 * rolling_std

    # Calculate Bollinger Bandwidth as the normalized difference between upper and lower bands
    # This shows how wide the bands are relative to the average price
    bandwidth = (upper - lower) / np.where(rolling_mean != 0, rolling_mean, np.nan)

    # Return the bandwidth series (proportional width of Bollinger Bands)
    return bandwidth


def calculate_cumulative_spike(data, period, threshold):
    # Calculate a binary spike indicator based on cumulative returns over a rolling window

    # Ensure that returns are calculated in the dataframe; calculate simple returns if missing
    if 'returns' not in data:
        data['returns'] = data['close'].pct_change()

    # Use log returns (log(1 + return)) to stabilize compounding effects for numerical accuracy
    log_returns = np.log1p(data['returns'])

    # Calculate rolling sum of log returns over the specified period
    # min_periods=period ensures full window before calculating
    # Shift by 1 to only look at past returns (avoid peeking into future)
    cumulative_return = (
        log_returns
        .rolling(window=period, min_periods=period)
        .sum()
        .shift(1)
        # Convert log returns sum back to linear scale using exponentiation and subtract 1
        .pipe(lambda x: np.exp(x) - 1)
        # Convert decimal to percentage scale (e.g., 0.05 → 5%)
        .mul(100)
    )

    # Generate binary indicator: 1 if cumulative return >= threshold, else 0
    # Fill missing values with 0 (no spike)
    return (cumulative_return >= threshold).fillna(0).astype(int)


def calculate_cumulative_change(data, last_change_length):
    # Calculate cumulative percentage price change over a recent window length

    # Calculate returns if not already present
    if 'returns' not in data:
        data['returns'] = data['close'].pct_change()

    # Convert simple returns to log returns for numerical stability when summing
    log_returns = np.log1p(data['returns'])

    # Calculate rolling sum of log returns over the specified recent window length
    cumulative_log = (
        log_returns
        .rolling(window=last_change_length, min_periods=last_change_length)
        .sum()
        .shift(1)  # Shift by one period to avoid using current period's return
    )

    # Convert summed log returns back to normal returns (exponentiate and subtract 1)
    # Multiply by 100 to express as percentage
    # Fill missing values with 0 and convert to float type
    return (
            np.exp(cumulative_log) - 1
    ).mul(100).fillna(0).astype(float)


def calculate_keltner_breakout(data, ema_period, atr_period, multiplier, cumulative_limit):
    # Calculate Exponential Moving Average (EMA)
    data['ema'] = data['close'].ewm(span=ema_period, adjust=False).mean()
    # Calculate Average True Range (ATR) for volatility
    data['atr'] = calculate_atr(data, atr_period)

    # Calculate upper band of Keltner channel (EMA + multiplier * ATR)
    upper_band = data['ema'] + (multiplier * data['atr'])

    # Reference price shifted 8 periods ago for cumulative price change calculation
    reference_close = data['close'].shift(8)
    cumulative_change = ((data['close'] - reference_close) / reference_close.replace(0, 1)) * 100
    cumulative_change = cumulative_change.fillna(0)  # Fill NaNs with zero for stability

    # Check if price breaks upper band and cumulative price move is large enough
    is_breakout = (data['close'] > upper_band)
    has_significant_move = cumulative_change.abs() >= cumulative_limit

    # Combine conditions and exclude early rows where data insufficient
    breakouts = (
        (is_breakout & has_significant_move)
        .where(data.index >= max(ema_period, 4) + 1, False)
        .astype(int)  # Convert boolean to int 1/0
    )

    # Remove temporary columns to clean up dataframe
    data.drop(columns=['ema', 'atr'], inplace=True)

    return breakouts


def calculate_elder_ray_index(data, ema_period):
    # Calculate EMA for smoothing
    ema = data['close'].ewm(span=ema_period, adjust=False).mean()
    # Elder Ray Index = Close Price - EMA, indicates bullish or bearish pressure
    return data['close'] - ema


def calculate_atr(data_of_csv, period):
    # Calculate True Range components
    high_low = data_of_csv['high'] - data_of_csv['low']
    high_close = np.abs(data_of_csv['high'] - data_of_csv['close'].shift(1))
    low_close = np.abs(data_of_csv['low'] - data_of_csv['close'].shift(1))

    # True Range is max of the three ranges
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))

    # Calculate ATR as rolling average of True Range, skipping first element
    atr = true_range.rolling(window=period + 1).apply(
        lambda window: window[1:period + 1].sum() / period,
        raw=True
    )

    return atr


# 2. Dataset preparation
def prepare_sequences(data, features, window_size, preprocessor=None, outlier_window=60):
    data = data.copy()  # Avoid modifying original dataframe

    # Select numeric columns only (floats and ints) from features list
    numeric_features = data[features].select_dtypes(include=['float64', 'int64']).columns

    # Remove rows with missing numeric feature values
    if data[numeric_features].isna().any().any():
        data = data.dropna(subset=numeric_features)

    # Replace outliers for all features before scaling
    for feature in numeric_features:
        # Calculate rolling median for smoothing outlier values
        rolling_median = (
            data[feature]
            .rolling(window=outlier_window, min_periods=1, closed='left')
            .median()
        )

        # Calculate rolling Interquartile Range (IQR) for outlier detection
        q1 = data[feature].rolling(window=outlier_window, min_periods=1, closed='left').quantile(0.25)
        q3 = data[feature].rolling(window=outlier_window, min_periods=1, closed='left').quantile(0.75)
        iqr = q3 - q1

        # Define lower and upper bounds for normal data range
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Identify outliers outside bounds
        outlier_mask = (data[feature] < lower_bound) | (data[feature] > upper_bound)
        # Replace outliers with rolling median values to reduce noise
        data.loc[outlier_mask, feature] = rolling_median[outlier_mask]

    # Create or apply preprocessor (scaler)
    if preprocessor is None:
        preprocessor = ColumnTransformer(
            transformers=[('scaler', MinMaxScaler(), numeric_features)]
        )
        # Fit scaler on data and transform features to [0,1]
        scaled_data = preprocessor.fit_transform(data[features])
    else:
        # Use existing scaler to transform data (validation/test data)
        scaled_data = preprocessor.transform(data[features])

    # Extract target labels, aligned to start after window size due to sequence creation
    label = data['target'].values[window_size:]

    feature = []
    # Create sliding windows of data sequences for temporal model input
    for i in range(window_size, len(data)):
        feature.append(scaled_data[i - window_size:i])

    return np.array(feature), np.array(label), preprocessor


# 3. Model architecture construction
def build_model(input_shape):
    # Regularizer to keep the model in check — prevents overfitting by adding a small penalty on weights
    l2s = regularizers.L2(0.0001)

    # Input layer: flexible sequence length and number of features, ready to accept time-series data
    inputs = Input(shape=input_shape)

    # ========== CNN Layers — Feature Extractors 🚀 ==========
    # First convolution block: 64 filters scanning over time, picking out local patterns in the sequence
    x = Conv1D(64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2s)(inputs)
    x = BatchNormalization()(x)  # Keep activations smooth and stable — trains faster and better
    x = MaxPooling1D(pool_size=2, padding='same')(x)  # Halve the sequence length to focus on important features

    # Second convolution block: going deeper with 128 filters, catching even richer patterns
    x = Conv1D(128, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2s)(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)

    # ========== LSTM Layer — The Memory Keeper 🧠 ==========
    # Now the sequence goes through an LSTM to catch temporal dependencies — patterns over time
    x = LSTM(64, return_sequences=False, kernel_regularizer=l2s, recurrent_activation="tanh")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)  # Randomly drops 30% of neurons — prevents overfitting, makes the model robust

    # ========== Dense Layers — The Decision Makers 🎯 ==========
    # Fully connected dense layer with 64 neurons to combine learned features
    x = Dense(64, activation='relu', kernel_regularizer=l2s)(x)
    x = Dropout(0.3)(x)  # Another dropout for extra regularization

    # Final output layer: single neuron with sigmoid activation for binary classification (spike or no spike)
    outputs = Dense(1, activation='sigmoid', dtype='float32')(x)

    # Put it all together into a Keras Model object
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model:
    # - Adam optimizer for smooth and adaptive training
    # - Binary crossentropy loss because we’re doing binary classification
    # - Metrics to monitor: accuracy, precision, recall, and AUC — cover all angles of performance
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

    # feature categories from Java
    features_list = [
        'sma_crossover', 'trix', 'roc', 'bollinger_bands', 'cumulative_spike',
        'cumulative_change', 'keltner_breakout', 'elder_ray_index'
    ]

    # Define split index (80% train, 20% validation) to keep order
    split_index = int(len(data_of_csv) * 0.8)

    # Split data while maintaining chronological order
    train_data = data_of_csv.iloc[:split_index]
    val_data = data_of_csv.iloc[split_index:]

    # Verify size
    print(f"Training Data: {train_data.shape}, Labels: {train_data['target'].values.shape}")
    print(f"Validation Data: {val_data.shape}, Labels: {val_data['target'].values.shape}")

    # Scale the data to values between 0 and 1 (train, validation separate)
    features_train_scaled, labels_train_scaled, scaler = prepare_sequences(train_data, features_list, 28)
    features_test_scaled, labels_test_scaled, _ = prepare_sequences(val_data, features_list, 28, preprocessor=scaler)

    # Build and define the architecture of the Network for training (timeWindowSize, features)
    model = build_model((features_train_scaled.shape[1], features_train_scaled.shape[2]))

    # Define callbacks
    early_stop = EarlyStopping(monitor='val_auc', patience=20, mode='max', restore_best_weights=True, min_delta=0.005)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

    # Reshape data for SMOTE
    n_samples, window_size, n_features = features_train_scaled.shape

    # Flatten 3D array into 2D for SMOTE input (samples, features)
    features_train_2d = features_train_scaled.reshape(n_samples, window_size * n_features)

    # Apply SMOTE to balance classes: use dirty synthetic sample generation to trick
    # the model into the belief that everything which doesn't go up directly, since my dataset is made by hand
    # and only positive samples are selected, is bad. So by creating synthetic classes it believes that it has to
    # spike as soon as something goes up. Too high smote will bias the model so 0.5 - 0.7 works the best
    smote = SMOTE(sampling_strategy=0.7, random_state=42)

    # Fit SMOTE on training data and resample to balance classes
    features_train_smote_2d, labels_train_smote = smote.fit_resample(features_train_2d, labels_train_scaled)

    # Reshape back to 3D for model input (samples, time_steps, features)
    features_train_smote = features_train_smote_2d.reshape(-1, window_size, n_features)

    # Print class distributions before and after resampling
    print("Resampled class distribution:", np.bincount(labels_train_smote.flatten()))
    print("Class distribution:", np.bincount(labels_train_scaled.flatten()))

    # training process of the model CPU has in my case better performance than GPU
    # Explanation: tasks are moved around from cpu to gpu vice versa. Sometimes tasks are split between both which
    # introduces the constant of latency between communication which is 20ms per step vs 10 ms
    # since the architecture and the wide variety of components of this network don't allow
    # execution on one EP only efficient TensorFlow splits the process up but that isn't effective
    # hybrid models with assigning certain tasks forceful to EPs did improve the performance by 25% to 15 ms.
    with tf.device('/CPU:0'):
        model.fit(
            features_train_smote, labels_train_smote,
            epochs=50,
            batch_size=64,
            validation_data=(features_test_scaled, labels_test_scaled),
            callbacks=[early_stop, reduce_lr]
        )

    # Convert and save the ONNX model directly & define the input signature dynamically based on training data shape
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=[
        tf.TensorSpec([None, *features_train_scaled.shape[1:]], tf.float32)])
    with open("spike_predictor.onnx", "wb") as f:
        f.write(model_proto.SerializeToString())


if __name__ == "__main__":
    # Reproducibility settings
    SEED = 42
    # Fix Python hash seed for consistency in hashing operations
    os.environ['PYTHONHASHSEED'] = str(SEED)
    # Make TensorFlow operations deterministic for reproducible results
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Seed numpy RNG
    np.random.seed(SEED)
    # Seed TensorFlow RNG
    tf.random.set_seed(SEED)

    # Disable oneDNN optimizations to avoid possible inconsistencies
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    # Disable GPU devices to speed up training on CPU (as explained in comments)
    tf.config.set_visible_devices([], 'GPU')

    # Start the training process with data from CSV file
    train_spike_predictor('highFrequencyStocks.csv')
    print("Training done!")
