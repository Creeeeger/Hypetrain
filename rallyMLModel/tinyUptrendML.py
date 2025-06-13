import random
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
import tf2onnx
from keras.losses import BinaryCrossentropy
from keras.metrics import Precision, Recall, AUC
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D, MaxPool1D, LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam

# ====================
# Configuration / Parameters
# ====================
# Data & features
FEATURES = ['close', 'ma_10', 'slope_5', 'ret_1']
WINDOW_SIZE = 18
SPLIT_RATIO = 0.8
STOCK_FILE = "uptrendStocksQUBTUNAMBG.csv"
TEST_FILE = "uptrendStocksOBTSUNAMBG.csv"

# Random seed
SEED = 300277411

# Undersampling
NEG_SAMPLE_KEEP_RATIO = 1 - 0.94  # keep 6% of negatives

# Model hyperparameters
MODEL_PARAMS = {
    'conv_filters': 16,
    'conv_kernel_size': 2,
    'lstm_units': 32,
    'dropout_rate': 0.4,
    'dense_units': 16,
    'learning_rate': 1e-3
}

# Training settings
TRAIN_PARAMS = {
    'epochs': 20,
    'batch_size': 16,
    'es_patience': 5,
    'rlrp_factor': 0.5,
    'rlrp_patience': 3
}

# ONNX export
ONNX_OPSET = 18

# GPU/CPU
USE_GPU = False


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, on_bad_lines='skip')


def split_data(df: pd.DataFrame, ratio: float):
    split_idx = int(len(df) * ratio)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    return train_df, val_df


def make_sequences(df: pd.DataFrame):
    df['ma_10'] = df['close'].rolling(10, min_periods=1).mean()

    def slope(x):
        idx = np.arange(len(x))
        return np.polyfit(idx, x, 1)[0]

    df['slope_5'] = df['close'].rolling(5).apply(slope, raw=True).fillna(0)
    df['ret_1'] = df['close'].pct_change().fillna(0)

    arr = df[FEATURES].values
    X = [arr[i - WINDOW_SIZE:i] for i in range(WINDOW_SIZE, len(df))]
    y = df['target'].values[WINDOW_SIZE:]
    return np.array(X), np.array(y)


def build_power_model(seq_len: int, n_feats: int):
    inp = Input((seq_len, n_feats), name='input')
    x = Conv1D(
        filters=MODEL_PARAMS['conv_filters'],
        kernel_size=MODEL_PARAMS['conv_kernel_size'],
        padding='same', activation='relu')
    x = x(inp)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)
    x = LSTM(MODEL_PARAMS['lstm_units'])(x)
    x = Dropout(MODEL_PARAMS['dropout_rate'])(x)
    x = Dense(MODEL_PARAMS['dense_units'], activation='relu')(x)
    out = Dense(1, activation='sigmoid', name='out')(x)

    model = Model(inp, out, name='conv_lstm')
    model.compile(
        optimizer=Adam(MODEL_PARAMS['learning_rate']),
        loss=BinaryCrossentropy(),
        metrics=['accuracy', Precision(name='prec'), Recall(name='recall'), AUC(name='auc')]
    )
    return model


def export_to_onnx(model: Model, onnx_filename: str):
    output_name = model.outputs[0].name.split(':')[0]
    model.output_names = [output_name]

    input_signature = [
        tf.TensorSpec(shape=(None, WINDOW_SIZE, len(FEATURES)), dtype=model.inputs[0].dtype, name='input')
    ]
    onnx_model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature, opset=ONNX_OPSET
    )
    with open(onnx_filename, "wb") as f:
        f.write(onnx_model_proto.SerializeToString())
    print(f"ONNX model saved to {onnx_filename}")


if __name__ == "__main__":
    # Set seeds
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    print(f"Seed used for this run: {SEED}")

    if not USE_GPU:
        tf.config.set_visible_devices([], "GPU")

    # Load and prepare data
    df = load_data(STOCK_FILE)
    train_df, val_df = split_data(df, SPLIT_RATIO)
    X_train, y_train = make_sequences(train_df)
    X_val, y_val = make_sequences(val_df)

    print("TRAIN dist before:", Counter(y_train))

    # Under-sample negatives
    neg_idx = np.where(y_train == 0)[0]
    drop_size = int(len(neg_idx) * (1 - NEG_SAMPLE_KEEP_RATIO))
    drop_idx = np.random.choice(neg_idx, size=drop_size, replace=False)
    keep_mask = np.ones(len(y_train), dtype=bool)
    keep_mask[drop_idx] = False
    X_train, y_train = X_train[keep_mask], y_train[keep_mask]
    print("TRAIN dist after:", Counter(y_train))

    # Build and initialize model
    model = build_power_model(WINDOW_SIZE, X_train.shape[2])
    p = np.mean(y_train)
    b0 = np.log(p / (1 - p))
    model.layers[-1].bias.assign([b0])
    print(model.summary())

    # Training
    with tf.device('/CPU:0'):
        cw = class_weight.compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
        class_weight_dict = {0: cw[0], 1: cw[1]}

        es = EarlyStopping(monitor='val_auc', mode='max', patience=TRAIN_PARAMS['es_patience'],
                           restore_best_weights=True)
        rlrp = ReduceLROnPlateau(monitor='val_loss', factor=TRAIN_PARAMS['rlrp_factor'],
                                 patience=TRAIN_PARAMS['rlrp_patience'], verbose=1)

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=TRAIN_PARAMS['epochs'],
            batch_size=TRAIN_PARAMS['batch_size'],
            class_weight=class_weight_dict,
            callbacks=[es, rlrp]
        )

    # Print layer biases
    print("\n=== Layer biases ===")
    for layer in model.layers:
        if hasattr(layer, 'bias') and layer.bias is not None:
            print(f"{layer.name:20s} bias values: {layer.bias.numpy()}\n")

    # Evaluate on test set if provided
    if TEST_FILE:
        test_df = load_data(TEST_FILE)
        X_test, y_test = make_sequences(test_df)
        results = model.evaluate(X_test, y_test, batch_size=TRAIN_PARAMS['batch_size'])
        print("Test set evaluation:", dict(zip(model.metrics_names, results)))

        # Plot predictions vs close
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(12, 5))
        time_idx = test_df.index[WINDOW_SIZE:]
        ax1.plot(test_df['close'], 'b-', label='Close')
        ax2 = ax1.twinx()
        ax2.plot(time_idx, model.predict(X_test).ravel(), 'r-', label='Predicted P(up)')
        ax2.set_ylim(0, 1)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.show()

    # Export model
    export_to_onnx(model, "tinyUptrend.onnx")
