import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tf2onnx
from keras.losses import BinaryCrossentropy
from keras.metrics import Precision, Recall, AUC
from keras.src.layers import Concatenate, Lambda
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, MaxPool1D, LSTM, Dropout, Dense, RepeatVector, \
    TimeDistributed
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ====================
# Configuration / Parameters
# ====================
# Data & features
FEATURES = ['close', 'ma_10', 'slope_5', 'ret_1']
WINDOW_SIZE = 18
SPLIT_RATIO = 0.8
STOCK_FILE = "uptrendStocksQUBTUNAMBG.csv"
TEST_FILE = "uptrendStocksQUBTUNAMBG.csv"

# Random seed
SEED = int.from_bytes(os.urandom(4), 'big')

# Undersampling
NEG_SAMPLE_KEEP_RATIO = 1 - 0.94  # keep 6% of negatives

# Model hyperparameters
MODEL_PARAMS = {
    'conv_filters': 16,
    'conv_kernel_size': 2,
    'lstm_units': 32,
    'dropout_rate': 0.4,
    'dense_units': 8,
    'learning_rate': 1e-3
}

# Training settings
TRAIN_PARAMS = {
    'epochs': 50,
    'batch_size': 16,
    'es_patience': 5,
    'rlrp_factor': 0.5,
    'rlrp_patience': 3
}

AE_PARAMS = {
    "latent_dim": 32,  # size of bottleneck
    "lr": 1e-3,
    "epochs": 60,
    "batch_size": 128,
    "es_patience": 8,
    "rlrp_factor": 0.5,
    "rlrp_patience": 4,
    "noise_std": 0.15,  # Gaussian noise in latent space
    "synth_target_pos_ratio": 0.8  # aim for pos to be ~80% of negs after synth (tune)
}

# ONNX export
ONNX_OPSET = 18


def set_seeds():
    # Set seeds
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    print(f"Seed used for this run: {SEED}")


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, on_bad_lines='skip')


def split_data(df: pd.DataFrame, ratio: float):
    split_idx = int(len(df) * ratio)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    return train_df, val_df


def make_sequences(
        df: pd.DataFrame,
        min_pos_count: int = 5,
        require_consecutive: bool = True
):
    df = df.copy()

    # --- feature engineering ---
    df['ma_10'] = df['close'].rolling(10, min_periods=1).mean()

    def slope(x):
        idx = np.arange(len(x))
        return np.polyfit(idx, x, 1)[0]

    df['slope_5'] = df['close'].rolling(5).apply(slope, raw=True).fillna(0)
    df['ret_1'] = df['close'].pct_change().fillna(0)

    arr = df[FEATURES].values
    y_all = df['target'].values.astype(int)

    def max_consecutive_ones(a):
        # runs of ones length
        # e.g., [1,1,0,1] -> max 2
        if a.size == 0:
            return 0
        # Trick: differences of padded array to find run breaks
        padded = np.concatenate(([0], a, [0]))
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        run_lengths = ends - starts
        return int(run_lengths.max()) if run_lengths.size else 0

    X, y = [], []
    for i in range(WINDOW_SIZE, len(df)):
        window_feats = arr[i - WINDOW_SIZE:i].copy()
        window_labels = y_all[i - WINDOW_SIZE:i]

        # --- normalize per feature within this window ---
        mins = window_feats.min(axis=0)
        maxs = window_feats.max(axis=0)
        ranges = np.where(maxs - mins == 0, 1e-6, maxs - mins)
        window_feats = (window_feats - mins) / ranges

        # --- window label based on labels inside the window ---
        if require_consecutive:
            label = 1 if max_consecutive_ones(window_labels) >= min_pos_count else 0
        else:
            label = 1 if window_labels.sum() >= min_pos_count else 0

        X.append(window_feats)
        y.append(label)

    return np.array(X), np.array(y, dtype=int)


def build_model(seq_len: int, n_feats: int):
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


def build_seq_autoencoder(seq_len: int, n_feats: int, latent_dim: int):
    # ----- Encoder -----
    ae_inp = Input((seq_len, n_feats), name='ae_input')
    x = Conv1D(32, 5, padding='same', activation='relu', name='enc_conv1')(ae_inp)
    x = BatchNormalization(name='enc_bn1')(x)
    # (optional) REMOVE pooling to keep resolution (or switch to stride in conv)
    # x = MaxPool1D(2, name='enc_pool1')(x)
    x = LSTM(64, return_sequences=False, name='enc_lstm')(x)
    z = Dense(latent_dim, name='latent')(x)

    # ----- Decoder with positional ramp -----
    latent_input = Input((latent_dim,), name='latent_input')
    y = RepeatVector(seq_len, name='dec_repeat')(latent_input)

    # build a [seq_len, 1] ramp and tile for the batch, then concat with y
    pos = Lambda(lambda _:
        tf.tile(tf.linspace(0.0, 1.0, seq_len)[None, :, None], [tf.shape(_)[0], 1, 1]),
        name='time_ramp')(latent_input)
    y = Concatenate(name='dec_concat')([y, pos])           # shape: (B, T, latent_dim+1)

    y = LSTM(64, return_sequences=True, name='dec_lstm')(y)
    y = TimeDistributed(Dense(32, activation='relu'), name='dec_td1')(y)
    ae_out = TimeDistributed(Dense(n_feats), name='dec_out')(y)

    autoencoder = Model(ae_inp, ae_out, name='seq_autoencoder')
    # reuse decoder layers by name
    dy = autoencoder.get_layer('dec_repeat')(latent_input)
    pos2 = autoencoder.get_layer('time_ramp')(latent_input)
    dy = autoencoder.get_layer('dec_concat')([dy, pos2])
    dy = autoencoder.get_layer('dec_lstm')(dy)
    dy = autoencoder.get_layer('dec_td1')(dy)
    dec_out = autoencoder.get_layer('dec_out')(dy)
    decoder = Model(latent_input, dec_out, name='seq_decoder')

    encoder = Model(ae_inp, z, name='seq_encoder')
    autoencoder.compile(optimizer=Adam(AE_PARAMS["lr"]), loss=MeanSquaredError())
    return autoencoder, encoder, decoder


def build_seq_autoencoder1(seq_len: int, n_feats: int, latent_dim: int):
    # ----- Encoder -----
    ae_inp = Input((seq_len, n_feats), name='ae_input')
    x = Conv1D(32, 5, padding='same', activation='relu', name='enc_conv1')(ae_inp)
    x = BatchNormalization(name='enc_bn1')(x)
    x = MaxPool1D(2, name='enc_pool1')(x)
    x = LSTM(64, return_sequences=False, name='enc_lstm')(x)
    z = Dense(latent_dim, name='latent')(x)

    # ----- Decoder (tied to encoder via shared layers) -----
    y = RepeatVector(seq_len, name='dec_repeat')(z)
    y = LSTM(64, return_sequences=True, name='dec_lstm')(y)
    y = TimeDistributed(Dense(32, activation='relu'), name='dec_td1')(y)
    ae_out = TimeDistributed(Dense(n_feats), name='dec_out')(y)  # linear output (targets are normalized)

    autoencoder = Model(ae_inp, ae_out, name='seq_autoencoder')
    encoder = Model(ae_inp, z, name='seq_encoder')

    # Standalone decoder that accepts latent vectors
    latent_input = Input((latent_dim,), name='latent_input')
    dy = autoencoder.get_layer('dec_repeat')(latent_input)
    dy = autoencoder.get_layer('dec_lstm')(dy)
    dy = autoencoder.get_layer('dec_td1')(dy)
    dec_out = autoencoder.get_layer('dec_out')(dy)
    decoder = Model(latent_input, dec_out, name='seq_decoder')

    autoencoder.compile(optimizer=Adam(AE_PARAMS["lr"]), loss=MeanSquaredError())
    return autoencoder, encoder, decoder


# ====================
# Visualization helpers
# ====================

def _sample_indices(n_total: int, n: int):
    n = int(min(max(n, 1), n_total))
    return np.random.choice(n_total, size=n, replace=False)


def plot_windows(X: np.ndarray, feature_names, n: int = 5, title: str = "Windows"):
    """Plot n windows (each as a small multi-feature time series panel).
    X shape: (N, seq_len, n_feats). Values are expected to be window-normalized in [0,1]."""
    if X is None or X.size == 0:
        print(f"[PLOT] No data provided for '{title}'.")
        return

    n = min(n, X.shape[0])
    idx = _sample_indices(X.shape[0], n)

    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        w = X[idx[i]]  # (seq_len, n_feats)
        for f in range(w.shape[1]):
            ax.plot(w[:, f], label=str(feature_names[f]) if feature_names is not None else f"f{f}")
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"win {idx[i]}")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("t (steps)")
    if feature_names is not None:
        axes[0].legend(loc='upper right', ncol=min(4, w.shape[1]))
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def synthesize_positive_windows(encoder, decoder, X_pos, n_synth: int, noise_std: float):
    if n_synth <= 0:
        return np.empty((0, X_pos.shape[1], X_pos.shape[2]))

    # encode all positives
    Z = encoder.predict(X_pos, verbose=0)

    # sample with replacement, add noise, decode
    idx = np.random.randint(0, Z.shape[0], size=n_synth)
    Z_samp = Z[idx]
    Z_noisy = Z_samp + np.random.normal(0.0, noise_std, size=Z_samp.shape)
    X_synth = decoder.predict(Z_noisy, verbose=0)
    return X_synth


def train_autoencoder_on_positives(X_train, y_train, X_val=None, y_val=None):
    assert X_train.ndim == 3, f"Expected X_train with shape (N,T,C), got {X_train.shape}"
    seq_len, n_feats = X_train.shape[1], X_train.shape[2]
    autoencoder, encoder, decoder = build_seq_autoencoder(seq_len, n_feats, AE_PARAMS["latent_dim"])

    # Select positives
    X_pos = X_train[y_train == 1]
    if X_pos.shape[0] < 20:
        print(f"[AE] Too few positive windows ({X_pos.shape[0]}) — skipping AE training.")
        return None, None, None

    # Optional validation on positives (if provided and compatible)
    val_data = None
    if X_val is not None and y_val is not None:
        if X_val.ndim == 3 and X_val.shape[1:] == X_train.shape[1:]:
            if np.any(y_val == 1):
                X_pos_val = X_val[y_val == 1]
                if X_pos_val.shape[0] > 0:
                    val_data = (X_pos_val, X_pos_val)
        else:
            print(f"[AE] Skipping validation: X_val shape {X_val.shape} incompatible with X_train {X_train.shape}.")

    es = EarlyStopping(monitor='val_loss' if val_data else 'loss',
                       mode='min', patience=AE_PARAMS['es_patience'],
                       restore_best_weights=True)
    rlrp = ReduceLROnPlateau(monitor='val_loss' if val_data else 'loss',
                             factor=AE_PARAMS['rlrp_factor'],
                             patience=AE_PARAMS['rlrp_patience'],
                             verbose=1)

    autoencoder.fit(
        X_pos, X_pos,
        validation_data=val_data,
        epochs=AE_PARAMS['epochs'],
        batch_size=AE_PARAMS['batch_size'],
        callbacks=[es, rlrp],
        verbose=1
    )
    return autoencoder, encoder, decoder


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
    # set seed for reproducibility
    set_seeds()

    # Load and prepare data
    df = load_data(STOCK_FILE)

    # split into train and validation sets
    train_df, val_df = split_data(df, SPLIT_RATIO)

    # engineer features and normalize window data
    X_train, y_train = make_sequences(train_df)
    X_val, y_val = make_sequences(val_df)

    # --- make close-only copies for AE ---
    # Find the column index of 'close' in FEATURES
    close_idx = FEATURES.index('close')

    Xc_train = X_train[:, :, [close_idx]]  # shape: (N, T, 1)
    Xc_val = X_val[:, :, [close_idx]]  # shape: (M, T, 1)

    # --- train AE on train positives only ---
    ae, enc, dec = train_autoencoder_on_positives(Xc_train, y_train, Xc_val, y_val)

    # --- NEW: decide how many positives to synthesize ---
    if enc is not None and dec is not None:
        pos_count = int(np.sum(y_train == 1))
        neg_count = int(np.sum(y_train == 0))
        target_pos = int(AE_PARAMS["synth_target_pos_ratio"] * neg_count)
        n_synth = max(0, target_pos - pos_count)

        if n_synth > 0 and pos_count > 0:
            X_pos = Xc_train[y_train == 1]
            X_synth = synthesize_positive_windows(enc, dec, X_pos, n_synth, AE_PARAMS["noise_std"])
            y_synth = np.ones((X_synth.shape[0],), dtype=y_train.dtype)

            # concat synthetic positives
            #X_train = np.concatenate([X_train, X_synth], axis=0)
            #y_train = np.concatenate([y_train, y_synth], axis=0)
            #print(f"[AE] Synthesized {X_synth.shape[0]} positive windows. "  f"Train size now: {X_train.shape[0]}")

            # --- Visualize: synthetic positive windows from AE ---
            if X_synth.shape[0] > 0:
                plot_windows(X_synth, FEATURES, n=min(10, X_synth.shape[0]), title="Synthetic positive windows (AE)")
        else:
            print("[AE] No synthesis needed or unavailable.")
    else:
        print("[AE] Autoencoder unavailable; skipping synthesis.")

    # Under-sample negatives
    neg_idx = np.where(y_train == 0)[0]
    drop_size = int(len(neg_idx) * (1 - NEG_SAMPLE_KEEP_RATIO))
    drop_idx = np.random.choice(neg_idx, size=drop_size, replace=False)
    keep_mask = np.ones(len(y_train), dtype=bool)
    keep_mask[drop_idx] = False
    X_train, y_train = X_train[keep_mask], y_train[keep_mask]

    # Build and initialize model
    model = build_model(WINDOW_SIZE, X_train.shape[2])
    p = np.mean(y_train)
    b0 = np.log(p / (1 - p + 1e-8))
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
    export_to_onnx(model, "uptrendML.onnx")
