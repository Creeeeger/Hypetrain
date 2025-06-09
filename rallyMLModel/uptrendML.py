import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tf2onnx
from keras import Model, Input
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import Conv1D, BatchNormalization, Activation, Bidirectional, MultiHeadAttention, \
    LayerNormalization, LSTM
from keras.src.metrics import Recall, Precision, AUC
from numba import njit
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers import MaxPool1D, Dropout, Dense, Add

# pip install command:
# pip install numpy pandas scikit-learn numba tensorflow tf2onnx keras

# ——————————————
# CONFIGURATION
# ——————————————
DATA_PATH = "uptrendStocks.csv"
DATA_PATH_TEST = "uptrendStocks3.csv"
WINDOW_SIZE = 30
OUTLIER_WINDOW = 300
AUG_TIMES = 5
SPLIT_RATIO = 0.8
SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
SLOPE_WINDOWS = [5, 10, 20]
FEATURE_COLUMNS = [
                      'open', 'high', 'low', 'close', 'volume', 'pct_change', 'ma_10', 'std_10'
                  ] + [
                      f'close_slope_{w}' for w in SLOPE_WINDOWS  # multi-bar slopes
                  ] + [
                      f'momentum_{w}' for w in SLOPE_WINDOWS  # multi-bar momentum
                  ] + [
                      'volatility', 'avg_volume'#, 'fund_trend', 'gap_down', 'wicks_ok', 'micro_up', 'near_res'
                  ]

TARGET_COLUMN = 'target'
ONNX_FILENAME = "uptrendPredictor.onnx"


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
        start = 0 if i - window + 1 < 0 else i - window + 1
        length = i - start + 1
        Sx = 0.0
        Sy = 0.0
        Sxy = 0.0
        Sxx = 0.0
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


def compute_fundamental_trend(df: pd.DataFrame,
                              window: int,
                              min_change_pct: float,
                              min_green_ratio: float) -> pd.Series:
    close = df['close']
    open_ = df['open']

    first = close.shift(window)
    last = close
    pct_change = (last - first) / first * 100

    is_green = (close > open_).astype(int)
    green_ratio = is_green.rolling(window).mean()

    red_size = (open_ - close).clip(lower=0)
    max_red = red_size.rolling(window).max()

    total_gain = (last - first).clip(lower=0)
    no_big_pullback = max_red < (total_gain / 3)

    return ((pct_change >= min_change_pct) &
            (green_ratio >= min_green_ratio) &
            no_big_pullback).astype(int)


def compute_gap_down(df: pd.DataFrame, tolerance_pct: float) -> pd.Series:
    prev_close = df['close'].shift(1)
    return (df['open'] + prev_close * (tolerance_pct / 100) < prev_close).astype(int)


def compute_bad_wicks_ok(df: pd.DataFrame,
                         bars_to_check: int,
                         wick_tol: float) -> pd.Series:
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    closes = df['close'].values
    n = len(df)
    out = np.zeros(n, dtype=int)

    for i in range(n):
        start = i - bars_to_check + 1
        if start < 0:
            continue
        bad = 0
        ok = True
        for j in range(start, i + 1):
            rng = highs[j] - lows[j]
            if rng <= 0:
                bad += 1
                continue
            low_w = (opens[j] - lows[j]) / rng
            high_w = (highs[j] - closes[j]) / rng
            if low_w > 0.7 or high_w > 0.7:
                ok = False
                break
            if low_w > wick_tol or high_w > wick_tol:
                bad += 1
        out[i] = int(ok and (bad <= bars_to_check // 3))
    return pd.Series(out, index=df.index)


def compute_micro_up(df: pd.DataFrame,
                     bars_to_check: int,
                     flat_tol_pct: float,
                     min_pump_pct: float) -> pd.Series:
    close = df['close'].values
    n = len(df)
    out = np.zeros(n, dtype=int)

    for i in range(n):
        start = i - bars_to_check + 1
        if start < 0:
            continue
        flats = 0
        pump = False
        for j in range(start, i):
            prev, curr = close[j], close[j + 1]
            if curr < prev * (1 + flat_tol_pct / 100):
                flats += 1
            elif curr >= prev * (1 + min_pump_pct / 100):
                pump = True
        out[i] = int(pump and (flats <= 1))
    return pd.Series(out, index=df.index)


def compute_near_resistance(df: pd.DataFrame,
                            lookback: int,
                            tol_pct: float = 0.5) -> pd.Series:
    highs = df['high'].values
    close = df['close'].values
    n = len(df)
    out = np.zeros(n, dtype=int)

    for i in range(n):
        if i < 1:
            continue
        start = max(0, i - lookback)
        res = highs[start:i].max()
        out[i] = int((close[i] >= res * (1 - tol_pct / 100)) and (close[i] <= res))
    return pd.Series(out, index=df.index)


# ——————————————
# 3) PREPARE SEQUENCES
# ——————————————
def prepare_sequences(data: pd.DataFrame, features: list[str], target: str, window_size: int, outlier_window: int,
                      aug_times: int = 0, preprocessor=None):
    df = data.copy()

    # ————————————
    # 1) feature engineering
    # ————————————
    df['pct_change'] = df['close'].pct_change().fillna(0)
    df['ma_10'] = df['close'].rolling(10, min_periods=1).mean()
    df['std_10'] = df['close'].rolling(10, min_periods=1).std().fillna(0)

    close_arr = df['close'].values.astype(np.float64)
    for w in SLOPE_WINDOWS:
        df[f'close_slope_{w}'] = compute_slopes_numba(close_arr, w)
        df[f'momentum_{w}'] = df['close'].pct_change(w).fillna(0) * 100.0

    max_w = max(SLOPE_WINDOWS)
    df['volatility'] = df['close'].rolling(max_w, min_periods=1).std().fillna(0)
    df['avg_volume'] = df['volume'].rolling(max_w, min_periods=1).mean().fillna(0)

  #  df['fund_trend'] = compute_fundamental_trend(df, window=window_size, min_change_pct=0.5, min_green_ratio=0.6)
   # df['gap_down'] = compute_gap_down(df, tolerance_pct=0.2)
   # df['wicks_ok'] = compute_bad_wicks_ok(df, bars_to_check=5, wick_tol=0.3)
   # df['micro_up'] = compute_micro_up(df, bars_to_check=5, flat_tol_pct=0.1, min_pump_pct=0.5)
   # df['near_res'] = compute_near_resistance(df, lookback=15, tol_pct=0.5)

    # ————————————
    # 2) outlier smoothing
    # ————————————
    numeric = features
    for feat in numeric:
        rm = df[feat].rolling(window=outlier_window, min_periods=1, closed='left').median()
        q1 = df[feat].rolling(window=outlier_window, min_periods=1, closed='left').quantile(0.05)
        q3 = df[feat].rolling(window=outlier_window, min_periods=1, closed='left').quantile(0.95)
        iqr = q3 - q1
        lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (df[feat] < lb) | (df[feat] > ub)
        df.loc[mask, feat] = rm[mask]

    # ————————————
    # 3) build raw windows (no scaling yet)
    # ————————————
    raw_X, raw_y = [], df[target].values[window_size:]
    arr = df[features].values
    for i in range(window_size, len(df)):
        raw_X.append(arr[i - window_size:i])

    raw_X = np.array(raw_X)  # shape = (n_windows, window_size, n_features)
    raw_y = np.array(raw_y)

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
        scaler = preprocessor.named_transformers_['scaler']
        scale_vals = scaler.scale_
        min_vals = scaler.min_

        # Print Java float array for SCALE
        print("public static final float[] SCALE = new float[]{")
        for v in scale_vals:
            print(f"    {v}F,")
        print("};\n")

        # Print Java float array for MIN_OFFSET
        print("public static final float[] MIN_OFFSET = new float[]{")
        for v in min_vals:
            print(f"    {v}F,")
        print("};")
    else:
        scaled_flat = preprocessor.transform(flat)

    X_scaled = scaled_flat.reshape(n_samples, seq_len, n_feats)

    return X_scaled, raw_y, preprocessor


# ——————————————
# 4) DATA AUGMENTATION
# ——————————————
def scale_price_range(window, min_f=0.9, max_f=1.1):
    f = np.random.uniform(min_f, max_f)
    w = window.copy()
    w[:, :4] *= f
    return w


def shift_price_range(window, max_s=2.0):
    s = np.random.uniform(-max_s, max_s)
    w = window.copy()
    w[:, :4] += s
    return w


def distort_volume_and_change(window, vol_s=0.001):
    w = window.copy()
    w[:, 4] *= np.random.uniform(1 - vol_s, 1 + vol_s)
    return w


def jitter_time_series(window, sigma=0.001):
    return window + np.random.normal(0, sigma, size=window.shape)


def augment_window(window):
    w = scale_price_range(window)
    w = shift_price_range(w)
    w = distort_volume_and_change(w)
    w = jitter_time_series(w)
    return w


def augment_data(X, y, times: int):
    minority_idx = np.where(y == 1)[0]

    X_aug, y_aug = [], []
    for i in minority_idx:
        for _ in range(times):
            X_aug.append(augment_window(X[i]))
            y_aug.append(1)

    if len(X_aug) == 0:
        # no minority samples to augment
        return X, y

    X_full = np.vstack([X, np.array(X_aug)])
    y_full = np.concatenate([y, np.array(y_aug)])
    return X_full, y_full


# ——————————————
# 5) MODEL BUILDING & TRAINING
# ——————————————
def build_model(seq_len: int, n_features: int):
    inp = tf.keras.Input(shape=(seq_len, n_features), name="input")

    # 1D convolution to pick up local slope patterns
    x = tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(inp)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(64, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # LSTM to capture longer-term dependencies
    x = tf.keras.layers.LSTM(64, return_sequences=False)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # A little dense head
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(x)

    return tf.keras.Model(inp, out)


# ——————————————
# 5) MODEL WITH DILATED CNN + BiLSTM + BN + Focal Loss
# ——————————————
def build_dilated_CNN(seq_len: int, n_feats: int):
    inp = tf.keras.Input((seq_len, n_feats), name='input')
    x = inp
    for rate in [1, 2]:
        x = tf.keras.layers.Conv1D(32, 3, padding='same', dilation_rate=rate)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Bidirectional(LSTM(64, return_sequences=False))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid', name='out')(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer='adam',
        loss=binary_focal_loss(alpha=0.25, gamma=2.0),
        metrics=['accuracy', Precision(), Recall(), AUC()]
    )
    return model


def build_attention_head_model(seq_len: int, n_feats: int,
                              d_model=64, num_heads=4, ff_dim=128,
                              conv_filters=32, lstm_units=64):
    inp = Input(shape=(seq_len, n_feats), name="input")

    # 1) Local feature extractor
    x = tf.keras.layers.Conv1D(conv_filters, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.Conv1D(conv_filters, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool1D(2)(x)  # now length = seq_len/2

    # 2) Project to d_model
    x = tf.keras.layers.Dense(d_model)(x)

    # 3) Transformer encoder block
    # 3a) Self‐attention
    attn_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=0.1
    )(x, x)
    attn_output = tf.keras.layers.Dropout(0.1)(attn_output)
    x = tf.keras.layers.Add()([x, attn_output])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    # 3b) Feed-forward
    ff = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    ff = tf.keras.layers.Dense(d_model)(ff)
    ff = tf.keras.layers.Dropout(0.1)(ff)
    x = tf.keras.layers.Add()([x, ff])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    # 4) Bidirectional LSTM to capture order
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=False))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # 5) Head
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    return Model(inputs=inp, outputs=out)


def binary_focal_loss(alpha=0.25, gamma=2.0):
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
        # Get predicted probabilities
        y_pred_prob = self.model.predict(self.X_val, verbose=0).reshape(-1)
        y_pred = (y_pred_prob >= 0.5).astype(int)
        f1 = f1_score(self.y_val, y_pred)
        print(f"\nval_f1: {f1:.4f}")
        logs = logs or {}
        logs["val_f1"] = f1


def train_model(model: tf.keras.Model, X_train, y_train, X_val, y_val, epochs: int, batch_size: int):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy',
                           Precision(name='precision'),
                           Recall(name='recall'),
                           AUC(name='auc')])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    f1_callback = F1Metrics(X_val, y_val)

    return model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            reduce_lr,
            f1_callback],
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


def plot_confusion_matrix(model, X, y_true, threshold=0.8, labels=('0', '1')):
    # 1) Get predicted probabilities and hard labels
    cls_probs = model.predict(X, verbose=0).reshape(-1)
    y_pred = (cls_probs >= threshold).astype(int)

    # 2) Text confusion matrix + classification report
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cr = classification_report(y_true, y_pred, digits=4)
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print("           pred=0    pred=1")
    print(f"true=0   {cm[0, 0]:8d}   {cm[0, 1]:8d}")
    print(f"true=1   {cm[1, 0]:8d}   {cm[1, 1]:8d}")

    print("\nClassification Report:\n")
    print(cr)

    # 3) Graphic confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    ax.set_title(f'Confusion Matrix (thr={threshold})')
    plt.show()

    return cm, cr


# ——————————————
# MAIN EXECUTION
# ——————————————
if __name__ == "__main__":
    set_reproducibility(SEED)

    df = load_data(DATA_PATH)

    train_df, val_df = split_data(df, SPLIT_RATIO)

    # 1) figure out how many aug rounds to equalize each split
    cnt_train = Counter(train_df[TARGET_COLUMN])
    times_train = int(np.ceil(cnt_train[0] / cnt_train[1])) - 1

    cnt_val = Counter(val_df[TARGET_COLUMN])
    times_val = int(np.ceil(cnt_val[0] / cnt_val[1])) - 1

    # 2) prepare & augment *before* scaling
    X_train, y_train, scaler = prepare_sequences(
        train_df,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        WINDOW_SIZE,
        OUTLIER_WINDOW,
        aug_times=times_train,
        preprocessor=None
    )

    X_val, y_val, _ = prepare_sequences(
        val_df,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        WINDOW_SIZE,
        OUTLIER_WINDOW,
        aug_times=0,
        preprocessor=scaler
    )

    df_test = load_data(DATA_PATH_TEST)
    train_test_df, val_test_df = split_data(df_test, SPLIT_RATIO)

    X_test, y_test, _ = prepare_sequences(
        df_test,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        WINDOW_SIZE,
        OUTLIER_WINDOW,
        aug_times=0,
        preprocessor=scaler
    )

    val_counts = Counter(y_val)
    total = len(y_val)
    print({cls: f"{cnt} ({cnt / total:.1%})" for cls, cnt in val_counts.items()})

    # 3) shuffle
    perm = np.random.permutation(len(y_train))
    X_train, y_train = X_train[perm], y_train[perm]

    perm_val = np.random.permutation(len(y_val))
    X_val, y_val = X_val[perm_val], y_val[perm_val]

    with tf.device('/CPU:0'):
        model = build_dilated_CNN(WINDOW_SIZE, X_train.shape[2])

        print(model.summary())

        train_model(
            model,
            X_train, y_train,
            X_val, y_val,
            EPOCHS,
            BATCH_SIZE
        )

    print("Training complete!")

    export_to_onnx(model, ONNX_FILENAME)

    plot_confusion_matrix(model, X_test, y_test, threshold=0.8)
