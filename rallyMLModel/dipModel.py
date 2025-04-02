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
from sklearn.preprocessing import MinMaxScaler


# 1. Feature Engineering
def create_features(data_of_csv):
    data_of_csv = data_of_csv.copy()
    data_of_csv['returns'] = data_of_csv['close'].pct_change(fill_method=None)

    # Add RSI for oversold detection
    data_of_csv['rsi'] = calculate_rsi(data_of_csv, 14)

    # Add Volume Spike detection
    data_of_csv['volume_spike'] = calculate_volume_spike(data_of_csv)

    # Add support/resistance levels
    data_of_csv['support_level'] = calculate_support_level(data_of_csv)

    data_of_csv['target'] = data_of_csv['target'].astype(int)

    data_of_csv.dropna(inplace=True)

    return data_of_csv


def calculate_rsi(data, period):
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_volume_spike(data):
    return (data['volume'] > data['volume'].rolling(20).mean() * 2).astype(int)


def calculate_support_level(data):
    return data['low'].rolling(20).min()


# 2. Dataset preparation
def prepare_sequences(data, features, window_size, preprocessor=None, outlier_window=60):
    data = data.copy()
    numeric_features = data[features].select_dtypes(include=['float64', 'int64']).columns

    # Check for NaN values and drop rows with missing values
    if data[numeric_features].isna().any().any():
        data = data.dropna(subset=numeric_features)

    # Outlier replacement for ALL features first
    for feature in numeric_features:
        rolling_median = (
            data[feature]
            .rolling(window=outlier_window, min_periods=1, closed='left')
            .median()
        )

        q1 = data[feature].rolling(window=outlier_window, min_periods=1, closed='left').quantile(0.25)
        q3 = data[feature].rolling(window=outlier_window, min_periods=1, closed='left').quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (data[feature] < lower_bound) | (data[feature] > upper_bound)
        data.loc[outlier_mask, feature] = rolling_median[outlier_mask]

    if preprocessor is None:
        preprocessor = ColumnTransformer(
            transformers=[('scaler', MinMaxScaler(), numeric_features)]
        )
        # Apply the preprocessor to the data
        scaled_data = preprocessor.fit_transform(data[features])
    else:
        # Apply the preprocessor to the data
        scaled_data = preprocessor.transform(data[features])  # Use existing scaler

    label = data['target'].values[window_size:]

    feature = []
    # Create window arrays of the data for historical processing
    for i in range(window_size, len(data)):
        feature.append(scaled_data[i - window_size:i])

    return np.array(feature), np.array(label), preprocessor


# 3. Model architecture construction
def build_model(input_shape):
    l2s = regularizers.L2(0.0001)
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
                           AUC(name='auc')
                           ]
                  )
    return model


# 4. Trainings loop
def train_spike_predictor(data_path_train, data_path_val):
    # read the data of the CSV file and skip bad lines to prevent errors
    train_data = pd.read_csv(data_path_train, on_bad_lines='skip')
    val_data = pd.read_csv(data_path_val, on_bad_lines='skip')

    # Engineer the features
    train_data = create_features(train_data)
    val_data = create_features(val_data)

    # feature categories from Java
    features_list = [
        ''
    ]

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
    features_train_2d = features_train_scaled.reshape(n_samples, window_size * n_features)

    smote = SMOTE(sampling_strategy=0.7, random_state=42)
    features_train_smote_2d, labels_train_smote = smote.fit_resample(features_train_2d, labels_train_scaled)

    # Reshape back to 3D for model input
    features_train_smote = features_train_smote_2d.reshape(-1, window_size, n_features)

    # Update class weight calculation using resampled data
    print("Resampled class distribution:", np.bincount(labels_train_smote.flatten()))
    print("Class distribution:", np.bincount(labels_train_scaled.flatten()))

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
    with open("dip_predictor.onnx", "wb") as f:
        f.write(model_proto.SerializeToString())


if __name__ == "__main__":
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    tf.config.set_visible_devices([], 'GPU')

    train_spike_predictor('dipStocksTrain.csv', 'dipStocksVal.csv')  # Main function for training
    print("Training done!")

'''
Okay, no I need the mother to inform me as precise as possible. That means we don't need to look into the future. after we dipped like two or 3% the recovery 
might not start instantaneously but with a shift of five minutes as an example. My main model is not capable of catching these upward trends, but this model 
needs to pin out the point when they recovery starts
'''
