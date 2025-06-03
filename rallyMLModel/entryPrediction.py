import json

import numpy as np
import tensorflow as tf
import tf2onnx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
JSON_PATH = "notifications.json"  # one JSON object per line
LOOKBACK_SIZE = 30  # we now accept ≥ 30, but only use last 30
VALIDATE_SIZE = 10  # must be at least 10
FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume', 'percentageChange']
THRESHOLD_PCT = 0.012  # 1.2% breakout criterion


def load_notifications_from_jsonl(json_path):
    """
    Assumes `json_path` contains one single JSON array of notification objects.
    Each object must have:
      - "lookbackWindow": a list of ≥ LOOKBACK_SIZE bars (we’ll use exactly the last LOOKBACK_SIZE).
      - "validationWindow": a list of ≥ VALIDATE_SIZE bars (we’ll use the first VALIDATE_SIZE for labeling).
    """
    with open(json_path, 'r') as f:
        all_notifs = json.load(f)  # parse the entire array

    windows = []
    labels = []
    skipped = 0

    for obj in all_notifs:
        lookback = obj.get("lookbackWindow", [])
        val_bars = obj.get("validationWindow", [])

        # 1) require lookback length ≥ LOOKBACK_SIZE
        if len(lookback) < LOOKBACK_SIZE:
            skipped += 1
            continue

        # 2) require validation length ≥ VALIDATE_SIZE
        if len(val_bars) < VALIDATE_SIZE:
            skipped += 1
            continue

        # take exactly the last LOOKBACK_SIZE bars
        recent_lookback = lookback[-LOOKBACK_SIZE:]

        # build a (30×6) array
        x_window = np.zeros((LOOKBACK_SIZE, len(FEATURE_COLS)), dtype=float)
        for i, bar in enumerate(recent_lookback):
            x_window[i, 0] = bar['open']
            x_window[i, 1] = bar['high']
            x_window[i, 2] = bar['low']
            x_window[i, 3] = bar['close']
            x_window[i, 4] = bar['volume']
            x_window[i, 5] = bar['percentageChange']

        # label from first VALIDATE_SIZE bars of validationWindow
        future_closes = np.array([vb['close'] for vb in val_bars[:VALIDATE_SIZE]], dtype=float)
        alert_close = x_window[-1, FEATURE_COLS.index('close')]
        future_peak = future_closes.max()
        label = 1 if (future_peak >= alert_close * (1 + THRESHOLD_PCT)) else 0

        windows.append(x_window)
        labels.append(label)

    print(f"Loaded {len(windows)} valid notifications, skipped {skipped}")
    if not windows:
        raise RuntimeError("No valid notifications—check JSON or size constraints.")
    x = np.stack(windows, axis=0)  # (N_valid, 30, 6)
    y = np.array(labels, dtype=int)  # (N_valid,)
    return x, y


# -------------------------------------------------------------
# 2. SCALING & SPLITTING
# -------------------------------------------------------------
def scale_and_split(X, y, test_size=0.2, random_state=42):
    n, seq_len, n_feat = X.shape
    flat = X.reshape(n * seq_len, n_feat)
    scaler = StandardScaler()
    flat_scaled = scaler.fit_transform(flat)
    x_scaled = flat_scaled.reshape(n, seq_len, n_feat)

    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  Train samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")
    return x_train, x_test, y_train, y_test, scaler


# -------------------------------------------------------------
# 3. MODEL ARCHITECTURE
# -------------------------------------------------------------
def create_lstm_model(seq_len, n_features, lr=1e-3):
    model = Sequential([
        LSTM(64, input_shape=(seq_len, n_features), return_sequences=False, recurrent_activation="tanh"),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=lr),
        metrics=['accuracy']
    )

    return model


# -------------------------------------------------------------
# 4. TRAIN + EVALUATE
# -------------------------------------------------------------
def train_and_evaluate(X_train, y_train, X_test, y_test,
                       validation_split=0.2, epochs=15, batch_size=32):
    seq_len, n_feat = X_train.shape[1], X_train.shape[2]

    model = create_lstm_model(seq_len, n_feat)
    print(model.summary())

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )
    loss, acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"\n► Final Test Loss: {loss:.4f} | Test Accuracy: {acc:.4f}")
    return model, history


# -------------------------------------------------------------
# 6. MAIN
# -------------------------------------------------------------
if __name__ == "__main__":
    # A) Load X, y (now accepts any lookbackWindow length ≥ 30)
    X, y = load_notifications_from_jsonl(JSON_PATH)
    print("X shape:", X.shape)  # e.g. (185, 30, 6)
    print("y shape:", y.shape)  # e.g. (185,)

    # C) Scale + split
    X_train, X_test, y_train, y_test, scaler = scale_and_split(X, y)

    # D) Train + evaluate
    model, history = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        validation_split=0.2, epochs=15, batch_size=32
    )

    # E) Save model
    output_tensor = model.outputs[0]  # instead of model.output

    # 2. Strip off the “:0” suffix from its name
    output_name = output_tensor.name.split(':')[0]

    # 3. Assign output_names so tf2onnx conversion won’t crash
    model.output_names = [output_name]

    # 4. Now you can build the TensorSpec and convert to ONNX
    seq_len, n_features = X_train.shape[1], X_train.shape[2]
    spec = [tf.TensorSpec([None, seq_len, n_features], tf.float32, name="input")]
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)

    # 4. Write the ONNX bytes to disk:
    with open("entryPrediction.onnx", "wb") as f:
        f.write(model_proto.SerializeToString())

    # F) Test model
    new_window = np.array([
                              [26.50, 26.70, 26.45, 26.65, 30000, 0.15],
                              [26.60, 26.80, 26.55, 26.75, 32000, 0.10],
                          ] + list(np.random.rand(28, 6)), dtype=float)  # final shape should be (30, 6)

    # 1) Flatten and scale with the exact same `scaler` used during training:
    flat = new_window.reshape(30, 6)
    scaled_flat = scaler.transform(flat)  # `scaler` was returned by scale_and_split(...)
    new_scaled = scaled_flat.reshape(1, 30, 6)  # shape (1, 30, 6)

    # 2) Feed into the trained model:
    prob = model.predict(new_scaled)[0][0]
    print(f"Predicted probability of a 'good spike up': {prob:.4f}")
