import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tf2onnx
from keras import Model, Input
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.metrics import Recall, Precision, AUC
from numba import njit
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K
from tqdm import trange

# pip install command:
# pip install numpy pandas scikit-learn numba tensorflow tf2onnx keras shap scikit-learn tqdm

# ——————————————
# CONFIGURATION
# ——————————————
test_file = "stocksTEST.csv"
stock_file = "uptrendStocksOBTSUNAMBG.csv"
ONNX_FILENAME = "uptrendPredictor.onnx"
TARGET_COLUMN = 'target'

WINDOW_SIZE = 30
SPLIT_RATIO = 0.8
SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
STRIDE = 10
FEATURE_COLUMNS = [
    'close', 'pct_change', 'ma_10', 'std_10',
    'up3', 'roc3_pos', 'slope_3', 'slope_5', 'slope_7',
    'rsi_14', 'macd_line', 'macd_signal', 'atr_14', 'obv'
]


# ——————————————
# 1) SET UP REPRODUCIBILITY & CPU ONLY
# ——————————————
def set_reproducibility(seed: int):
    tf.config.set_visible_devices([], "GPU")
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ——————————————
# 2) LOAD & SPLIT DATA
# ——————————————
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, on_bad_lines='skip')


def split_data(df: pd.DataFrame, ratio: float):
    split_idx = int(len(df) * ratio)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    return train_df, val_df


@njit
def compute_slopes_numba(arr: np.ndarray, window: int) -> np.ndarray:
    n = arr.shape[0]
    slopes = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - window + 1)
        length = i - start + 1
        Sx = Sy = Sxy = Sxx = 0.0
        for k in range(length):
            x = float(k)
            y = arr[start + k]
            Sx += x
            Sy += y
            Sxy += x * y
            Sxx += x * x
        denom = length * Sxx - Sx * Sx
        slopes[i] = (length * Sxy - Sx * Sy) / denom if denom != 0.0 else 0.0
    return slopes


# ——————————————
# 3) PREPARE SEQUENCES
# ——————————————
def prepare_sequences(data: pd.DataFrame, features: list[str], target: str, window_size: int,
                      aug_times: int = 0, preprocessor=None, oversample=False):
    df = data.copy()

    # ————————————
    # 1) feature engineering
    # ————————————
    df['pct_change'] = df['close'].pct_change().fillna(0)
    df['ma_10'] = df['close'].rolling(10, min_periods=1).mean()
    df['std_10'] = df['close'].rolling(10, min_periods=1).std().fillna(0)

    df['up3'] = (
        (df['close'] > df['close'].shift(1))
        .astype(int)
        .rolling(window=3, min_periods=1)
        .sum()
        .eq(3)  # True only if sum==3
        .astype(int)  # 1 for True, 0 for False
    )

    df['roc3_pos'] = (
        df['close'].pct_change(3)
        .fillna(0)
        .gt(0)  # True if >0
        .astype(int)  # 1 for True, 0 for False
    )

    # slopes
    close_arr = df['close'].values
    df['slope_3'] = compute_slopes_numba(close_arr, 3)
    df['slope_5'] = compute_slopes_numba(close_arr, 5)
    df['slope_7'] = compute_slopes_numba(close_arr, 7)

    # RSI(14)
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14, min_periods=1).mean()
    roll_down = down.rolling(14, min_periods=1).mean()
    rs = roll_up / (roll_down + 1e-6)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df['macd_line'] = macd_line
    df['macd_signal'] = macd_signal

    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14, min_periods=1).mean()

    # OBV (needs volume)
    if 'volume' in df.columns:
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    else:
        df['obv'] = 0.0

    # ————————————
    # 3) build raw windows (no scaling yet)
    # ————————————
    raw_X, raw_y = [], df[target].values[window_size:]
    arr = df[features].values
    for i in range(window_size, len(df)):
        raw_X.append(arr[i - window_size:i])

    raw_X = np.array(raw_X)  # shape = (n_windows, window_size, n_features)
    raw_y = np.array(raw_y)

    if oversample:
        raw_X, raw_y = slide_oversample(raw_X, raw_y, WINDOW_SIZE, STRIDE)

    # ————————————
    # 4) augment BEFORE scaling
    # ————————————
    if aug_times > 0:
        raw_X, raw_y = augment_data(raw_X, raw_y, aug_times)

    # ————————————
    # 5) scale
    # ————————————
    n_samples, seq_len, n_feats = raw_X.shape
    flat = raw_X.reshape(-1, n_feats)
    if preprocessor is None:
        preprocessor = ColumnTransformer(
            [('scaler', MinMaxScaler(), list(range(n_feats)))]
        )
        scaled_flat = preprocessor.fit_transform(flat)
    else:
        scaled_flat = preprocessor.transform(flat)

    X_scaled = scaled_flat.reshape(n_samples, seq_len, n_feats)

    return X_scaled, raw_y, preprocessor


def slide_oversample(X, y, window, stride):
    X_new, y_new = [], []
    pos = np.where(y == 1)[0]
    for idx in pos:
        for shift in range(1, window, stride):
            if idx + shift < len(X):
                X_new.append(X[idx + shift])
                y_new.append(1)
    if not X_new:
        return X, y
    return np.vstack([X, np.array(X_new)]), np.concatenate([y, np.array(y_new)])


# balanced batch generator
def balanced_generator(X, y, batch_size):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    half = batch_size // 2
    while True:
        p = np.random.choice(pos, half, replace=True)
        n = np.random.choice(neg, half, replace=True)
        idx = np.concatenate([p, n])
        np.random.shuffle(idx)
        yield X[idx], y[idx]


# ——————————————
# 4) DATA AUGMENTATION
# ——————————————
def jitter_time_series(window, sigma=0.005):
    return window + np.random.normal(0, sigma, size=window.shape)


def augment_data(X, y, times: int):
    minority_idx = np.where(y == 1)[0]

    X_aug, y_aug = [], []
    for i in minority_idx:
        for _ in range(times):
            X_aug.append(jitter_time_series(X[i]))
            y_aug.append(1)

    if len(X_aug) == 0:
        # no minority samples to augment
        return X, y

    X_full = np.vstack([X, np.array(X_aug)])
    y_full = np.concatenate([y, np.array(y_aug)])
    return X_full, y_full


# ——————————————
# 5) MODEL
# ——————————————
def TCN_block(x, filters, kernel, dilation):
    prev = x
    x = tf.keras.layers.Conv1D(filters, kernel, padding='causal', dilation_rate=dilation)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.SpatialDropout1D(0.2)(x)
    if prev.shape[-1] != filters:
        prev = tf.keras.layers.Conv1D(filters, 1, padding='same')(prev)
    return tf.keras.layers.Add()([prev, x])


def build_tcn_model(window: int, n_feats: int):
    inp = Input((window, n_feats))
    x = inp
    for d in [1, 2, 4, 8]:
        x = TCN_block(x, 64, 3, d)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = Model(inp, out)
    model.compile(
        optimizer='adam',
        loss=binary_focal_loss(),
        metrics=['accuracy', Precision(), Recall(), AUC()]
    )
    return model


def build_dilated_CNN(seq_len: int, n_feats: int):
    inp = tf.keras.Input((seq_len, n_feats), name='input')
    x = inp

    x = tf.keras.layers.Conv1D(32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.MaxPool1D(2)(x)

    x = tf.keras.layers.LSTM(64, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

    out = tf.keras.layers.Dense(1, activation='sigmoid', name='out')(x)
    model = tf.keras.Model(inp, out)

    model.compile(
        optimizer='adam',
        loss=binary_focal_loss(),
        metrics=['accuracy', Precision(), Recall(), AUC()]
    )
    return model


def binary_focal_loss(alpha=0.75, gamma=2.0):
    def loss(y_true, y_pred):
        # clip to prevent NaNs
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        # compute cross‐entropy
        ce = - (y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        # weight term: alpha for positives, (1-alpha) for negatives
        weight_pos = alpha * K.pow(1 - y_pred, gamma)
        weight_neg = (1 - alpha) * K.pow(y_pred, gamma)
        weight = y_true * weight_pos + (1 - y_true) * weight_neg
        return K.mean(weight * ce, axis=-1)

    return loss


class F1Metrics(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # 1) get predicted probabilities
        y_pred_prob = self.model.predict(self.X_val, verbose=0).ravel()

        # drop any NaN or infinite preds
        mask = np.isfinite(y_pred_prob)
        if not mask.all():
            n_bad = np.sum(~mask)
            print(f"  ⚠️  dropping {n_bad} NaN/inf preds from threshold sweep")
            y_pred_prob = y_pred_prob[mask]
            y_true = self.y_val[mask]
        else:
            y_true = self.y_val

        # now safe to compute pr curve
        p, r, thr = precision_recall_curve(y_true, y_pred_prob)

        f1s = 2 * p * r / (p + r + 1e-8)
        best_ix = np.argmax(f1s)
        best_thr = thr[best_ix]
        best_f1 = f1s[best_ix]

        # 4) print & log
        print(f"\n → Best thresh: {best_thr:.3f}  |  Best F1: {best_f1:.4f}")
        logs["val_f1"] = best_f1
        logs["val_thr"] = best_thr


def train_model(model: tf.keras.Model, X_train, y_train, X_val, y_val, epochs: int, batch_size: int):
    reduce_lr = ReduceLROnPlateau(monitor='val_f1', mode='max', factor=0.5, patience=3, min_lr=1e-6)

    f1_callback = F1Metrics(X_val, y_val)

    steps = len(y_train) // BATCH_SIZE

    return model.fit(
        balanced_generator(X_train, y_train, BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=epochs,
        # batch_size=batch_size,
        steps_per_epoch=steps,
        callbacks=[
            f1_callback,
            reduce_lr,
            EarlyStopping(
                monitor='val_f1',
                mode='max',
                patience=10,
                restore_best_weights=True,
            ),
        ],
        verbose=1
    )


def export_to_onnx(model, onnx_filename: str):
    # Make sure the model has exactly one output
    output_name = model.outputs[0].name.split(':')[0]
    model.output_names = [output_name]

    # Build the input signature from our sequences
    seq_len = WINDOW_SIZE
    n_features = len(FEATURE_COLUMNS)
    input_signature = [
        tf.TensorSpec(
            shape=(None, seq_len, n_features),
            dtype=model.inputs[0].dtype,
            name='input'
        )
    ]

    # Convert and save
    onnx_model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature,
        opset=18
    )

    with open(onnx_filename, "wb") as f:
        f.write(onnx_model_proto.SerializeToString())
    print(f"ONNX model saved to {onnx_filename}")


# ────────────────────────────────────────────────────────────────────────────────
#  2) PERMUTATION IMPORTANCE (F1 DROP)
# ────────────────────────────────────────────────────────────────────────────────
def permutation_importance_f1(model, X_val, y_val, feature_names,
                              threshold=0.5, n_repeats=5, random_state=42):
    rng = np.random.default_rng(random_state)

    # Baseline F1
    base_probs = model.predict(X_val, verbose=0).ravel()
    base_pred = (base_probs >= threshold).astype(int)
    base_f1 = f1_score(y_val, base_pred)

    importances = np.zeros(len(feature_names))

    for f_idx in trange(len(feature_names), desc="Permutation"):
        delta = 0.0
        for _ in range(n_repeats):
            X_perm = X_val.copy()
            # Permute the feature across the *batch* (keeps temporal structure inside each window)
            X_perm[:, :, f_idx] = rng.permutation(X_perm[:, :, f_idx])
            probs = model.predict(X_perm, verbose=0).ravel()
            pred = (probs >= threshold).astype(int)
            perm_f1 = f1_score(y_val, pred)
            delta += base_f1 - perm_f1
        importances[f_idx] = delta / n_repeats

    return sorted(zip(feature_names, importances), key=lambda kv: kv[1], reverse=True)


def check_permutations(model, X_val, y_val, feature_names):
    print("\n=== Permutation F1 drop ===")
    perm_rank = permutation_importance_f1(model, X_val, y_val, FEATURE_COLUMNS, threshold=0.5, n_repeats=5)
    for feat, score in perm_rank[:20]:
        print(f"{feat:25s} {score:10.5f}")
    # Quick leak-check
    sus_feats = [f for f, _ in perm_rank[:10]]
    print("\nTop-10 permutation losers (suspect if they include 'shift', 'jitter', "
          "'volume'-only props):")
    print(sus_feats)


def plot_prediction(model):
    probs = model.predict(X_full, verbose=0).ravel()

    # 4) Align each probability with the last bar of its window (use df_test!)
    time_axis = df_test.index[WINDOW_SIZE:]
    # 5) Plot price vs. probability (also from df_test)
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Price line from df_test
    ax1.plot(df_test['close'], label='Smoothed Close', color='tab:blue')
    ax1.set_ylabel('Price', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Probability line
    ax2 = ax1.twinx()
    ax2.plot(time_axis, probs, label='Uptrend Prob.', color='tab:red', lw=2)
    ax2.set_ylabel('Probability', color='tab:red')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    plt.title("Smoothed Close vs. Uptrend Probability")
    plt.tight_layout()
    plt.show()


# ——————————————
# MAIN EXECUTION
# ——————————————
if __name__ == "__main__":
    set_reproducibility(SEED)

    # 1) Load your one stock
    df = load_data(stock_file)

    # 3) Split into train/val (optional if you’re retraining)
    train_df, val_df = split_data(df, SPLIT_RATIO)

    # 4) Prepare sequences & get scaler on the training half
    X_train, y_train, scaler = prepare_sequences(
        train_df,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        WINDOW_SIZE,
        aug_times=0,
        preprocessor=None,
        oversample=True
    )

    print("TRAIN dist before:", Counter(y_train))

    # ─── UNDER-SAMPLE NEGATIVES ───────────────────────────────────────────────────
    # drop 70% of the zero-labels to reduce class imbalance
    neg_idx = np.where(y_train == 0)[0]
    drop_idx = np.random.choice(neg_idx, size=int(len(neg_idx) * 0.9), replace=False)
    keep = np.ones(len(y_train), dtype=bool)
    keep[drop_idx] = False
    X_train, y_train = X_train[keep], y_train[keep]

    print("TRAIN dist after:", Counter(y_train))

    # 5) Prepare validation (or test) with the same scaler
    X_val, y_val, _ = prepare_sequences(
        val_df,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        WINDOW_SIZE,
        aug_times=0,
        preprocessor=scaler,
        oversample=True
    )
    print("VAL dist:", Counter(y_val))

    # 6) Train your model
    with tf.device('/CPU:0'):
        model = build_dilated_CNN(WINDOW_SIZE, X_train.shape[2])
        final_model = build_tcn_model(WINDOW_SIZE, X_train.shape[2])

        print(model.summary())
        print(final_model.summary())

        train_model(
            model,
            X_train, y_train,
            X_val, y_val,
            EPOCHS,
            BATCH_SIZE
        )

        train_model(
            final_model,
            X_train, y_train,
            X_val, y_val,
            EPOCHS,
            BATCH_SIZE
        )

    print("\n=== Layer biases ===")
    for layer in model.layers:
        # most layers store bias as layer.bias
        if hasattr(layer, 'bias') and layer.bias is not None:
            b = layer.bias.numpy()
            print(f"{layer.name:20s} bias shape {b.shape} → values:\n{b}\n")
        else:
            # some layers (e.g. pooling) have no bias
            print(f"{layer.name:20s} has no bias\n")

    print("\n=== Layer biases Final Model TNC ===")
    for layer in final_model.layers:
        # most layers store bias as layer.bias
        if hasattr(layer, 'bias') and layer.bias is not None:
            b = layer.bias.numpy()
            print(f"{layer.name:20s} bias shape {b.shape} → values:\n{b}\n")
        else:
            # some layers (e.g. pooling) have no bias
            print(f"{layer.name:20s} has no bias\n")

    # 7) Export to ONNX
    export_to_onnx(model, ONNX_FILENAME)

    check_permutations(model, X_val, y_val, FEATURE_COLUMNS)
    check_permutations(final_model, X_val, y_val, FEATURE_COLUMNS)

    df_test = load_data(test_file)

    # 8) For plotting, rebuild sequences on the full, smoothed series
    X_full, y_full, _ = prepare_sequences(
        df_test,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        WINDOW_SIZE,
        aug_times=0,
        preprocessor=scaler,
        oversample=False
    )

    plot_prediction(model)
    plot_prediction(final_model)
