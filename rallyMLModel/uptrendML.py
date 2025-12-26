import argparse
import os
import random

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(REPO_ROOT, ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(REPO_ROOT, ".cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import tf2onnx
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Precision, Recall, AUC
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    MaxPool1D,
    UpSampling1D,
    GlobalAveragePooling1D,
    LSTM,
    Dropout,
    Dense,
    Reshape,
    RepeatVector,
    TimeDistributed,
)
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
TARGET_COL = "target"
STOCK_FILE = "uptrendStocksQUBTUNAMBG__target_clean__minrun8_gr0.9_red1_ret0.002_gap0.csv"
TEST_FILE = "uptrendStocksOBTSUNAMBG__target_clean__minrun8_gr0.9_red1_ret0.002_gap0.csv"
SEQ_STRIDE = 1
NORMALIZATION = "zscore"  # "zscore" keeps sign; "minmax" matches old behavior

# Random seed
SEED = int.from_bytes(os.urandom(4), 'big')

# Negative sampling / balancing (training only)
NEG_POS_RATIO = 3.0  # keep at most this many negatives per positive (0=keep all negatives)
NEG_SAMPLING = "mixed"  # "random", "near_pos", "mixed"
NEAR_POS_RADIUS = 300  # sequences within +/- this distance from a positive sequence
HARD_NEG_FRACTION = 0.7  # for mixed: fraction of negatives sampled from near_pos bucket

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
    "latent_dim": 128,  # size of bottleneck
    "conv_filters": 96,
    "conv_kernel_size": 5,
    "lstm_units": 128,
    "lr": 1e-4,
    "epochs": 100,
    "batch_size": 128,
    "es_patience": 8,
    "rlrp_factor": 0.5,
    "rlrp_patience": 4,
    "diff_loss_weight": 3.0,
    "close_diff_weight": 10.0,
    "close_recon_weight": 5.0,
    "noise_std": 0.15,  # Gaussian noise in latent space
    "synth_target_pos_ratio": 0.8  # aim for pos to be ~80% of negs after synth (tune)
}

# ONNX export
ONNX_OPSET = 18


def set_seeds(seed: int | None = None) -> int:
    seed_used = int(SEED if seed is None else seed)
    random.seed(seed_used)
    np.random.seed(seed_used)
    tf.random.set_seed(seed_used)
    print(f"Seed used for this run: {seed_used}")
    return seed_used


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, on_bad_lines='skip')


def split_data(df: pd.DataFrame, ratio: float):
    split_idx = int(len(df) * ratio)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    return train_df, val_df


def normalize_window(window_feats: np.ndarray, method: str) -> np.ndarray:
    if method == "minmax":
        mins = window_feats.min(axis=0)
        maxs = window_feats.max(axis=0)
        ranges = np.where(maxs - mins == 0, 1e-6, maxs - mins)
        return (window_feats - mins) / ranges

    if method == "zscore":
        means = window_feats.mean(axis=0)
        stds = window_feats.std(axis=0)
        stds = np.where(stds == 0, 1e-6, stds)
        out = (window_feats - means) / stds
        return np.clip(out, -6.0, 6.0)

    if method == "none":
        return window_feats

    raise ValueError(f"Unknown normalization method: {method}")


def normalize_windows_batch(X: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return X
    out = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        out[i] = normalize_window(X[i], method)
    return out


def compute_global_minmax(X: np.ndarray):
    mins = X.min(axis=(0, 1))
    maxs = X.max(axis=(0, 1))
    return mins, maxs


def apply_global_minmax(X: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    ranges = np.where(maxs - mins == 0, 1e-6, maxs - mins)
    return (X - mins) / ranges


def invert_global_minmax(X: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    ranges = np.where(maxs - mins == 0, 1e-6, maxs - mins)
    return X * ranges + mins


def make_sequences(
        df: pd.DataFrame,
        min_pos_count: int = 7,
        require_consecutive: bool = True,
        target_col: str = TARGET_COL,
        stride: int = SEQ_STRIDE,
        normalization: str = NORMALIZATION,
        return_end_indices: bool = False,
        apply_normalization: bool = True,
):
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {df.columns.tolist()}")

    # --- feature engineering ---
    df['ma_10'] = df['close'].rolling(10, min_periods=1).mean()

    def slope_log(x):
        idx = np.arange(len(x))
        x = np.asarray(x, dtype=np.float64)
        x = np.log(np.clip(x, 1e-9, None))
        return np.polyfit(idx, x, 1)[0]

    # slope on log(close) is scale-invariant (approx. return per bar)
    df['slope_5'] = df['close'].rolling(5).apply(slope_log, raw=True).fillna(0)

    # log-return keeps sign and is price-scale invariant
    lc = np.log(np.clip(df['close'].astype(float).values, 1e-9, None))
    df['ret_1'] = pd.Series(lc).diff().fillna(0).to_numpy()

    arr = df[FEATURES].values
    y_all = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).values

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
    end_indices = []
    stride = int(max(1, stride))
    for i in range(WINDOW_SIZE, len(df), stride):
        window_feats = arr[i - WINDOW_SIZE:i].copy()
        window_labels = y_all[i - WINDOW_SIZE:i]

        # --- normalize per feature within this window (unless disabled) ---
        if apply_normalization:
            window_feats = normalize_window(window_feats, normalization)

        # --- window label based on labels inside the window ---
        if require_consecutive:
            label = 1 if max_consecutive_ones(window_labels) >= min_pos_count else 0
        else:
            label = 1 if window_labels.sum() >= min_pos_count else 0

        X.append(window_feats)
        y.append(label)
        end_indices.append(i - 1)

    X = np.array(X)
    y = np.array(y, dtype=int)
    if return_end_indices:
        return X, y, np.array(end_indices, dtype=int)
    return X, y


def downsample_negatives(
        X: np.ndarray,
        y: np.ndarray,
        neg_pos_ratio: float,
        sampling: str,
        near_pos_radius: int,
        hard_neg_fraction: float,
        seed: int,
):
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    if pos_idx.size == 0 or neg_pos_ratio <= 0:
        return X, y

    desired_neg = int(min(neg_idx.size, round(neg_pos_ratio * pos_idx.size)))
    if desired_neg >= neg_idx.size:
        return X, y

    rng = np.random.default_rng(seed)
    sampling = str(sampling)
    near_pos_radius = int(max(0, near_pos_radius))
    hard_neg_fraction = float(np.clip(hard_neg_fraction, 0.0, 1.0))

    def sample_from(candidates: np.ndarray, k: int) -> np.ndarray:
        if k <= 0 or candidates.size == 0:
            return np.array([], dtype=int)
        k = int(min(k, candidates.size))
        return rng.choice(candidates, size=k, replace=False)

    if sampling not in {"random", "near_pos", "mixed"}:
        raise ValueError("neg_sampling must be one of: random, near_pos, mixed")

    if sampling == "random" or near_pos_radius == 0:
        neg_keep = sample_from(neg_idx, desired_neg)
    else:
        # Build a boolean mask of sequence indices near any positive (difference array trick)
        n = y.shape[0]
        diff = np.zeros(n + 1, dtype=np.int32)
        for p in pos_idx:
            s = max(0, int(p) - near_pos_radius)
            e = min(n, int(p) + near_pos_radius + 1)
            diff[s] += 1
            diff[e] -= 1
        near = np.cumsum(diff[:-1]) > 0

        hard_candidates = neg_idx[near[neg_idx]]
        easy_candidates = neg_idx[~near[neg_idx]]

        if sampling == "near_pos":
            neg_keep = sample_from(hard_candidates, desired_neg)
            if neg_keep.size < desired_neg:
                neg_keep = np.concatenate([neg_keep, sample_from(easy_candidates, desired_neg - neg_keep.size)])
        else:
            k_hard = int(round(desired_neg * hard_neg_fraction))
            k_easy = desired_neg - k_hard
            hard_pick = sample_from(hard_candidates, k_hard)
            easy_pick = sample_from(easy_candidates, k_easy)
            # If one bucket is empty, fill from the other
            remaining = desired_neg - (hard_pick.size + easy_pick.size)
            if remaining > 0:
                pool = np.setdiff1d(neg_idx, np.concatenate([hard_pick, easy_pick]), assume_unique=False)
                filler = sample_from(pool, remaining)
                neg_keep = np.concatenate([hard_pick, easy_pick, filler])
            else:
                neg_keep = np.concatenate([hard_pick, easy_pick])

    keep_idx = np.concatenate([pos_idx, neg_keep])
    rng.shuffle(keep_idx)
    return X[keep_idx], y[keep_idx]


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
    x = Conv1D(
        AE_PARAMS["conv_filters"],
        AE_PARAMS["conv_kernel_size"],
        padding='same',
        activation='relu',
        name='enc_conv1'
    )(ae_inp)
    x = BatchNormalization(name='enc_bn1')(x)
    x = MaxPool1D(2, name='enc_pool1')(x)
    x = LSTM(AE_PARAMS["lstm_units"], return_sequences=False, name='enc_lstm')(x)
    z = Dense(latent_dim, name='latent')(x)

    # ----- Decoder (tied to encoder via shared layers) -----
    y = RepeatVector(seq_len, name='dec_repeat')(z)
    y = LSTM(AE_PARAMS["lstm_units"], return_sequences=True, name='dec_lstm')(y)
    y = TimeDistributed(Dense(AE_PARAMS["conv_filters"], activation='relu'), name='dec_td1')(y)
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


def build_seq_autoencoder_new(seq_len: int, n_feats: int, latent_dim: int):
    # ----- Encoder -----
    enc_in = Input((seq_len, n_feats), name='ae_input')
    x = Conv1D(
        AE_PARAMS["conv_filters"],
        AE_PARAMS["conv_kernel_size"],
        padding='same',
        activation='relu',
        name='enc_conv1'
    )(enc_in)
    x = BatchNormalization(name='enc_bn1')(x)
    # (no pooling to keep temporal resolution)
    # x = MaxPool1D(2, name='enc_pool1')(x)
    x = LSTM(AE_PARAMS["lstm_units"], return_sequences=False, name='enc_lstm')(x)
    z = Dense(latent_dim, name='latent')(x)
    encoder = Model(enc_in, z, name='seq_encoder')

    # ----- Decoder -----
    lat_in = Input((latent_dim,), name='latent_input')
    y = RepeatVector(seq_len, name='dec_repeat')(lat_in)

    y = LSTM(AE_PARAMS["lstm_units"], return_sequences=True, name='dec_lstm')(y)
    y = TimeDistributed(Dense(AE_PARAMS["conv_filters"], activation='relu'), name='dec_td1')(y)
    dec_out = TimeDistributed(Dense(n_feats), name='dec_out')(y)
    decoder = Model(lat_in, dec_out, name='seq_decoder')

    # ----- Autoencoder (shares encoder/decoder weights) -----
    ae_out = decoder(encoder(enc_in))
    autoencoder = Model(enc_in, ae_out, name='seq_autoencoder')
    autoencoder.compile(optimizer=Adam(AE_PARAMS["lr"]), loss=MeanSquaredError())
    return autoencoder, encoder, decoder


def build_seq_autoencoder_conv(seq_len: int, n_feats: int, latent_dim: int):
    close_idx = FEATURES.index('close')

    def diff_mse(y_true, y_pred):
        d_true = y_true[:, 1:, :] - y_true[:, :-1, :]
        d_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        return tf.reduce_mean(tf.square(d_true - d_pred))

    def combined_loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        diff = diff_mse(y_true, y_pred)
        close_true = y_true[:, :, close_idx]
        close_pred = y_pred[:, :, close_idx]
        close_recon_loss = tf.reduce_mean(tf.square(close_true - close_pred))
        close_diff = close_true[:, 1:] - close_true[:, :-1]
        close_diff_pred = close_pred[:, 1:] - close_pred[:, :-1]
        close_diff_loss = tf.reduce_mean(tf.square(close_diff - close_diff_pred))
        return (
                mse
                + AE_PARAMS["diff_loss_weight"] * diff
                + AE_PARAMS["close_recon_weight"] * close_recon_loss
                + AE_PARAMS["close_diff_weight"] * close_diff_loss
        )

    # ----- Encoder -----
    enc_in = Input((seq_len, n_feats), name='ae_input')
    x = Conv1D(AE_PARAMS["conv_filters"], AE_PARAMS["conv_kernel_size"],
               padding='same', activation='relu', name='enc_conv1')(enc_in)
    x = BatchNormalization(name='enc_bn1')(x)
    x = MaxPool1D(2, name='enc_pool1')(x)
    x = Conv1D(AE_PARAMS["conv_filters"], AE_PARAMS["conv_kernel_size"],
               padding='same', activation='relu', name='enc_conv2')(x)
    x = GlobalAveragePooling1D(name='enc_gap')(x)
    z = Dense(latent_dim, name='latent')(x)
    encoder = Model(enc_in, z, name='seq_encoder')

    # ----- Decoder -----
    lat_in = Input((latent_dim,), name='latent_input')
    y = Dense((seq_len // 2) * AE_PARAMS["conv_filters"], activation='relu', name='dec_dense')(lat_in)
    y = Reshape((seq_len // 2, AE_PARAMS["conv_filters"]), name='dec_reshape')(y)
    y = UpSampling1D(2, name='dec_upsample')(y)
    y = Conv1D(AE_PARAMS["conv_filters"], AE_PARAMS["conv_kernel_size"],
               padding='same', activation='relu', name='dec_conv1')(y)
    dec_out = Conv1D(n_feats, 1, padding='same', activation=None, name='dec_out')(y)
    decoder = Model(lat_in, dec_out, name='seq_decoder')

    # ----- Autoencoder -----
    ae_out = decoder(encoder(enc_in))
    autoencoder = Model(enc_in, ae_out, name='seq_autoencoder')
    autoencoder.compile(optimizer=Adam(AE_PARAMS["lr"]), loss=combined_loss)
    return autoencoder, encoder, decoder


# ====================
# Visualization helpers
# ====================

def _sample_indices(n_total: int, n: int):
    n = int(min(max(n, 1), n_total))
    return np.random.choice(n_total, size=n, replace=False)


def plot_windows(
        X: np.ndarray,
        feature_names,
        n: int = 5,
        title: str = "Windows",
        save_path: str | None = None,
):
    """Plot n windows (each as a small multi-feature time series panel).
    X shape: (N, seq_len, n_feats)."""
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
        lo = float(np.nanmin(w))
        hi = float(np.nanmax(w))
        pad = (hi - lo) * 0.1 if hi > lo else 1.0
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(f"win {idx[i]}")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("t (steps)")
    if feature_names is not None:
        axes[0].legend(loc='upper right', ncol=min(4, w.shape[1]))
    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
        plt.close(fig)
        print(f"[PLOT] Saved {save_path}")
    else:
        plt.show()


def plot_ae_reconstructions(
        autoencoder,
        X_pos,
        n: int = 4,
        title: str = "AE reconstructions (close)",
        save_path: str | None = None,
        feature_idx: int = 0,
        feature_label: str | None = None,
):
    if X_pos is None or X_pos.size == 0:
        print("[AE] No positive windows available for reconstruction plot.")
        return

    n = min(n, X_pos.shape[0])
    idx = _sample_indices(X_pos.shape[0], n)
    X_sample = X_pos[idx]
    X_recon = autoencoder.predict(X_sample, verbose=0)

    fig, axes = plt.subplots(n, 1, figsize=(10, 2.4 * n), sharex=True)
    if n == 1:
        axes = [axes]
    label = feature_label if feature_label is not None else f"feat {feature_idx}"
    for i, ax in enumerate(axes):
        ax.plot(X_sample[i, :, feature_idx], color='black', linewidth=1.5, label='orig')
        ax.plot(X_recon[i, :, feature_idx], color='tab:orange', linewidth=1.2, label='recon')
        ax.set_ylabel(label)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("t (steps)")
    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
        plt.close(fig)
        print(f"[PLOT] Saved {save_path}")
    else:
        plt.show()


def synthesize_positive_windows(
        encoder,
        decoder,
        X_pos,
        n_synth: int,
        noise_std: float,
        ae_scaler,
        norm_method: str,
):
    if n_synth <= 0:
        return np.empty((0, X_pos.shape[1], X_pos.shape[2]))

    # encode all positives
    Z = encoder.predict(X_pos, verbose=0)

    # sample with replacement, add noise, decode
    idx = np.random.randint(0, Z.shape[0], size=n_synth)
    Z_samp = Z[idx]
    Z_noisy = Z_samp + np.random.normal(0.0, noise_std, size=Z_samp.shape)
    X_synth_norm = decoder.predict(Z_noisy, verbose=0)
    ae_mins, ae_maxs = ae_scaler
    X_synth_raw = invert_global_minmax(X_synth_norm, ae_mins, ae_maxs)
    X_synth_normed = normalize_windows_batch(X_synth_raw, norm_method)
    return X_synth_normed


def train_autoencoder_on_positives(
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        plot_recon: bool = True,
        plot_dir: str | None = None,
        plot_n: int = 4,
        recon_n: int = 4,
):
    # Find the column index of 'close' in FEATURES
    close_idx = FEATURES.index('close')

    assert X_train.ndim == 3, f"Expected X_train with shape (N,T,C), got {X_train.shape}"
    seq_len, n_feats = X_train.shape[1], X_train.shape[2]

    # Global minmax on raw windows for AE stability
    ae_mins, ae_maxs = compute_global_minmax(X_train)
    X_train = apply_global_minmax(X_train, ae_mins, ae_maxs)
    if X_val is not None:
        X_val = apply_global_minmax(X_val, ae_mins, ae_maxs)
    autoencoder, encoder, decoder = build_seq_autoencoder_conv(seq_len, n_feats, AE_PARAMS["latent_dim"])

    print(autoencoder.summary())
    print(encoder.summary())
    print(decoder.summary())

    # Select positives
    X_pos = X_train[y_train == 1]
    if X_pos.shape[0] < 20:
        print(f"[AE] Too few positive windows ({X_pos.shape[0]}) — skipping AE training.")
        return None, None, None, None

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

    plot_windows(
        X_pos[:, :, [close_idx]],
        ['close'],
        n=min(plot_n, X_pos.shape[0]),
        title="positive windows (AE, global minmax)",
        save_path=f"{plot_dir}/ae_positive_windows.png" if plot_dir else None,
    )

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
    if plot_recon:
        plot_ae_reconstructions(
            autoencoder,
            X_pos,
            n=min(recon_n, X_pos.shape[0]),
            title="AE reconstructions (close)",
            save_path=f"{plot_dir}/ae_reconstructions.png" if plot_dir else None,
            feature_idx=close_idx,
            feature_label="close",
        )
    return autoencoder, encoder, decoder, (ae_mins, ae_maxs)


def export_to_onnx(model: Model, onnx_filename: str):
    if tf2onnx is None:
        print("[ONNX] tf2onnx not available; skipping export.")
        return
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


def features_from_close_batch(
        X_close: np.ndarray,
        ma_window: int = 10,
        slope_window: int = 5,
        normalization: str = NORMALIZATION,
) -> np.ndarray:
    if X_close.ndim == 2:
        X_close = X_close[:, :, None]

    N, T, _ = X_close.shape
    out = np.zeros((N, T, len(FEATURES)), dtype=np.float32)

    for i in range(N):
        c = X_close[i, :, 0].astype(np.float64)  # 1D (T,)

        # ma_10 with min_periods=1 (defined from first point)
        ma10 = pd.Series(c).rolling(ma_window, min_periods=1).mean().to_numpy()

        # slope_5 with variable-length window near the start; slope=0 if <2 points
        slopes = np.zeros(T, dtype=np.float64)
        for t in range(T):
            start = max(0, t - (slope_window - 1))
            seg = np.log(np.clip(c[start:t + 1], 1e-9, None))
            if seg.size >= 2:
                x = np.arange(seg.size, dtype=np.float64)
                s, _ = np.polyfit(x, seg, 1)  # slope
                slopes[t] = s
            else:
                slopes[t] = 0.0

        # ret_1: log return; 0 at first element
        ret = np.zeros(T, dtype=np.float64)
        lc = np.log(np.clip(c, 1e-9, None))
        ret[1:] = np.diff(lc)

        # Stack in FEATURE order
        W = np.stack([c, ma10, slopes, ret], axis=-1)  # (T, 4)

        W_norm = normalize_window(W, normalization)

        out[i] = W_norm.astype(np.float32)

    return out


def parse_args():
    p = argparse.ArgumentParser(description="Train uptrend model with separate train/eval files")
    p.add_argument("--train-file", default=STOCK_FILE, help="CSV used for training/validation split")
    p.add_argument("--eval-file", default=TEST_FILE, help="CSV used for evaluation (no split)")
    p.add_argument("--target-col", default=TARGET_COL, help="Target column name (e.g. target_clean)")
    p.add_argument("--split-ratio", type=float, default=SPLIT_RATIO, help="Train/val split ratio on train-file")
    p.add_argument("--seed", type=int, default=None, help="Fix seed for reproducible runs (default: random)")
    p.add_argument("--epochs", type=int, default=TRAIN_PARAMS["epochs"], help="Training epochs")
    p.add_argument("--batch-size", type=int, default=TRAIN_PARAMS["batch_size"], help="Batch size")
    p.add_argument("--stride", type=int, default=SEQ_STRIDE, help="Sequence stride (reduces overlap; e.g. 2,3,5)")
    p.add_argument(
        "--norm",
        choices=["zscore", "minmax"],
        default=NORMALIZATION,
        help="Per-window feature normalization method",
    )
    p.add_argument("--min-pos-count", type=int, default=7, help="Pos label requires >= this many positives in window")
    p.add_argument(
        "--no-require-consecutive",
        action="store_true",
        help="If set: use total positive count in window instead of consecutive run length",
    )
    p.add_argument(
        "--neg-pos-ratio",
        type=float,
        default=NEG_POS_RATIO,
        help="Training sampling: keep at most N negatives per positive (0=keep all)",
    )
    p.add_argument(
        "--neg-sampling",
        choices=["random", "near_pos", "mixed"],
        default=NEG_SAMPLING,
        help="How to pick negatives when downsampling",
    )
    p.add_argument(
        "--near-pos-radius",
        type=int,
        default=NEAR_POS_RADIUS,
        help="For near_pos/mixed: negatives within +/- this many sequences from any positive are considered 'hard'",
    )
    p.add_argument(
        "--hard-neg-fraction",
        type=float,
        default=HARD_NEG_FRACTION,
        help="For mixed: fraction of negatives sampled from near_pos bucket",
    )
    p.add_argument(
        "--use-ae",
        action="store_true",
        help="Enable autoencoder-based positive synthesis (recommended only with --norm=minmax)",
    )
    p.add_argument(
        "--ae-only",
        action="store_true",
        help="Train/visualize the autoencoder and exit (requires --use-ae)",
    )
    p.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory to save plots (if omitted, plots are shown interactively)",
    )
    p.add_argument(
        "--ae-plot-n",
        type=int,
        default=16,
        help="How many positive windows to plot for AE inspection",
    )
    p.add_argument(
        "--ae-recon-n",
        type=int,
        default=16,
        help="How many reconstruction samples to plot for AE inspection",
    )
    return p.parse_args()


if __name__ == "__main__":
    # set seed for reproducibility
    args = parse_args()
    seed_used = set_seeds(args.seed)
    require_consecutive = not args.no_require_consecutive
    if args.use_ae and args.norm not in {"minmax", "zscore"}:
        raise ValueError("--use-ae requires --norm=minmax or --norm=zscore.")
    if args.ae_only and not args.use_ae:
        raise ValueError("--ae-only requires --use-ae.")
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)

    # Load and prepare data
    df = load_data(args.train_file)

    # split into train and validation sets
    train_df, val_df = split_data(df, args.split_ratio)

    # engineer features and normalize window data
    X_train, y_train, _train_end_idx = make_sequences(
        train_df,
        target_col=args.target_col,
        min_pos_count=args.min_pos_count,
        require_consecutive=require_consecutive,
        stride=args.stride,
        normalization=args.norm,
        return_end_indices=True,
    )
    X_val, y_val = make_sequences(
        val_df,
        target_col=args.target_col,
        min_pos_count=args.min_pos_count,
        require_consecutive=require_consecutive,
        stride=1,
        normalization=args.norm,
    )

    if args.use_ae:
        X_train_raw, y_train_raw, _ = make_sequences(
            train_df,
            target_col=args.target_col,
            min_pos_count=args.min_pos_count,
            require_consecutive=require_consecutive,
            stride=args.stride,
            normalization="none",
            return_end_indices=True,
            apply_normalization=False,
        )
        X_val_raw, y_val_raw = make_sequences(
            val_df,
            target_col=args.target_col,
            min_pos_count=args.min_pos_count,
            require_consecutive=require_consecutive,
            stride=1,
            normalization="none",
            apply_normalization=False,
        )

        # --- train AE on train positives only ---
        ae, enc, dec, ae_scaler = train_autoencoder_on_positives(
            X_train_raw,
            y_train_raw,
            X_val_raw,
            y_val_raw,
            plot_dir=args.plot_dir,
            plot_n=args.ae_plot_n,
            recon_n=args.ae_recon_n,
        )
        if args.ae_only:
            print("[AE] --ae-only set; exiting after autoencoder training/visualization.")
            raise SystemExit(0)

        # --- decide how many positives to synthesize ---
        if enc is not None and dec is not None and ae_scaler is not None:
            pos_count = int(np.sum(y_train == 1))
            neg_count = int(np.sum(y_train == 0))
            target_pos = int(AE_PARAMS["synth_target_pos_ratio"] * neg_count)
            n_synth = max(0, target_pos - pos_count)

            if n_synth > 0 and pos_count > 0:
                X_synth = synthesize_positive_windows(
                    enc,
                    dec,
                    X_train_raw[y_train_raw == 1],
                    n_synth,
                    AE_PARAMS["noise_std"],
                    ae_scaler,
                    args.norm,
                )
                y_synth = np.ones((X_synth.shape[0],), dtype=y_train.dtype)

                X_train = np.concatenate([X_train, X_synth], axis=0)
                y_train = np.concatenate([y_train, y_synth], axis=0)
                print(f"[AE] Synthesized {X_synth.shape[0]} positive windows. Train size now: {X_train.shape[0]}")
            else:
                print("[AE] No synthesis needed or unavailable.")
        else:
            print("[AE] Autoencoder unavailable; skipping synthesis.")

    # Downsample negatives (keeps all positives)
    before = (int(np.sum(y_train == 1)), int(np.sum(y_train == 0)))
    X_train, y_train = downsample_negatives(
        X_train,
        y_train,
        neg_pos_ratio=args.neg_pos_ratio,
        sampling=args.neg_sampling,
        near_pos_radius=args.near_pos_radius,
        hard_neg_fraction=args.hard_neg_fraction,
        seed=seed_used,
    )
    after = (int(np.sum(y_train == 1)), int(np.sum(y_train == 0)))
    print(f"[Sampling] pos/neg before={before[0]}/{before[1]} after={after[0]}/{after[1]}")

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
            epochs=args.epochs,
            batch_size=args.batch_size,
            class_weight=class_weight_dict,
            callbacks=[es, rlrp]
        )

    # Print layer biases
    print("\n=== Layer biases ===")
    for layer in model.layers:
        if hasattr(layer, 'bias') and layer.bias is not None:
            print(f"{layer.name:20s} bias values: {layer.bias.numpy()}\n")

    # Evaluate on test set if provided
    if args.eval_file:
        test_df = load_data(args.eval_file)

        X_test, y_test = make_sequences(
            test_df,
            target_col=args.target_col,
            min_pos_count=args.min_pos_count,
            require_consecutive=require_consecutive,
            stride=1,
            normalization=args.norm,
        )

        results = model.evaluate(X_test, y_test, batch_size=args.batch_size)
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
