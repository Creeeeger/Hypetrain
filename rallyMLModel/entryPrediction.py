import json

import numpy as np
import tensorflow as tf
import tf2onnx
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.src.layers import MaxPooling1D, Conv1D
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

# Pip command: pip install numpy tensorflow tf2onnx scikit-learn keras

# ======================================
#           GLOBAL CONSTANTS
# ======================================

# File paths
JSON_PATH = "notifications.json"
VAL_JSON_PATH = "valnotifications.json"

# Window sizes
LOOKBACK_SIZE = 20
VALIDATE_SIZE = 3

# Feature columns and dimensions
FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume', 'percentageChange']
NUM_FEATURES = len(FEATURE_COLS)

# Model hyperparameters
LATENT_DIM = 32  # F
LSTM_UNITS = 16  # F
DROPOUT_RATE = 0.25  # F
ALPHA = 2.5  # F
LEARNING_RATE = 3e-3
EPOCHS = 50  # F
BATCH_SIZE = 16  # F

# Clustering / prototype extraction
N_BAD_PROTOTYPES = 15

# Train/test split
TEST_SPLIT = 0.2  # F
RANDOM_STATE = 42  # F

ONNX_FILENAME = "entryPrediction.onnx"  # F
TIMES_SIZE = 6  # 6F

# -------------  CNN hyper-params  -------------
CONV_FILTERS = 16
KERNEL_SIZE = 3
POOL_SIZE = 2


# ======================================
#           DATA LOADING
# ======================================

def load_notifications_from_jsonl(json_path):
    windows = []
    labels = []
    skipped = 0

    # Read newline-delimited JSON (one object per line)
    all_notifs = []
    with open(json_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                all_notifs.append(obj)
            except json.JSONDecodeError:
                # Skip malformed lines
                skipped += 1

    for obj in all_notifs:
        # 1) Grab the lookback‐ and validation‐window arrays
        lookback = obj.get("lookbackWindow", [])
        val_bars = obj.get("validationWindow", [])

        # 2) Skip if lookback or validation window is too short
        if len(lookback) < LOOKBACK_SIZE or len(val_bars) < VALIDATE_SIZE:
            skipped += 1
            continue

        # 3) Extract the provided 'target' label; skip if missing or not an integer
        raw_target = obj.get("target", None)
        if not isinstance(raw_target, int):
            skipped += 1
            continue
        label = raw_target

        # 4) Build the (LOOKBACK_SIZE × n_features) window from the last LOOKBACK_SIZE bars
        recent_lookback = lookback[-LOOKBACK_SIZE:]
        x_window = np.zeros((LOOKBACK_SIZE, len(FEATURE_COLS)), dtype=float)
        for i, bar in enumerate(recent_lookback):
            x_window[i, 0] = bar['open']
            x_window[i, 1] = bar['high']
            x_window[i, 2] = bar['low']
            x_window[i, 3] = bar['close']
            x_window[i, 4] = bar['volume']
            x_window[i, 5] = bar['percentageChange']

        # 5) Append to our lists
        windows.append(x_window)
        labels.append(label)

    print(f"Loaded {len(windows)} valid notifications, skipped {skipped}")
    if not windows:
        raise RuntimeError("No valid notifications—check JSON or size constraints.")

    x = np.stack(windows, axis=0)  # shape = (N, LOOKBACK_SIZE, n_features)
    y = np.array(labels, dtype=int)  # shape = (N,)

    return x, y


class F1Metrics(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        # Get predicted probabilities
        y_pred_prob = self.model.predict(self.X_val, verbose=0)[1].reshape(-1)
        y_pred = (y_pred_prob >= 0.5).astype(int)
        f1 = f1_score(self.y_val, y_pred)
        print(f"\nval_f1: {f1:.4f}")
        logs = logs or {}
        logs["val_f1"] = f1


# ======================================
#         SPLIT & SCALE DATA
# ======================================
def time_based_split_and_scale(X, y, test_size=TEST_SPLIT, random_state=42):
    """
    1) Randomly split X, y into train + test (stratified so labels remain balanced).
    2) Fit a StandardScaler on the _training_ windows (flattened).
    3) Return (scaled_train, scaled_test, y_train, y_test, scaler).
    """
    # 1) Random split (stratify by y to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 2) Fit scaler on flattened training sequence
    n_train, seq_len, n_feat = X_train.shape
    flat_train = X_train.reshape(n_train * seq_len, n_feat)
    scaler = StandardScaler().fit(flat_train)

    # 3) Scale train
    flat_train_scaled = scaler.transform(flat_train)
    X_train_scaled = flat_train_scaled.reshape(n_train, seq_len, n_feat)

    # 4) Scale test
    n_test = X_test.shape[0]
    flat_test = X_test.reshape(n_test * seq_len, n_feat)
    flat_test_scaled = scaler.transform(flat_test)
    X_test_scaled = flat_test_scaled.reshape(n_test, seq_len, n_feat)

    print(f"  Train samples: {X_train_scaled.shape[0]}, Test samples: {X_test_scaled.shape[0]}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ======================================
#       MODEL CONSTRUCTION
# ======================================

def build_seq_autoencoder_with_classifier(
        seq_len=LOOKBACK_SIZE,
        n_feat=NUM_FEATURES,
        latent_dim=LATENT_DIM,
        alpha=ALPHA,
        lr=LEARNING_RATE
):
    # ---------- INPUT ----------
    inp = Input(shape=(seq_len, n_feat), name="encoder_input")

    # ---------- NEW: temporal-conv front-end ----------
    x = Conv1D(filters=CONV_FILTERS, kernel_size=KERNEL_SIZE, padding="same", activation="relu", name="conv1d")(inp)
    x = MaxPooling1D(pool_size=POOL_SIZE, name="conv_pool")(x)
    x = Dropout(DROPOUT_RATE, name="conv_dropout")(x)

    # ---------- ORIGINAL ENCODER ----------
    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=False), name="enc_bi_lstm")(x)
    x = Dropout(DROPOUT_RATE, name="enc_dropout")(x)
    z = Dense(latent_dim, activation="linear", name="bottleneck")(x)

    # ---------- CLASSIFIER ----------
    cls = Dense(1, activation="sigmoid", name="classifier")(z)

    # ---------- DECODER ----------
    dec_in = RepeatVector(seq_len, name="repeat_z")(z)
    dec_lstm = LSTM(LSTM_UNITS, return_sequences=True, name="dec_lstm")(dec_in)
    dec_dropout = Dropout(DROPOUT_RATE, name="dec_dropout")(dec_lstm)
    recon = TimeDistributed(Dense(n_feat, activation="linear"), name="reconstruction")(dec_dropout)

    # ---------- COMPILE ----------
    model = tf.keras.Model(inputs=inp, outputs=[recon, cls], name="seq_autoenc_cnn_lstm")
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss={"reconstruction": MeanSquaredError(),
              "classifier": BinaryCrossentropy()},
        loss_weights={"reconstruction": 1.0, "classifier": alpha},
        metrics={
            "classifier": [
                "accuracy",
                tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
                tf.keras.metrics.AUC(name="pr_auc", curve="PR")
            ]
        }
    )

    # Separate encoder model for embedding extraction
    encoder_model = tf.keras.Model(inputs=inp, outputs=z, name="cnn_lstm_encoder")
    return model, encoder_model


# ======================================
#        TRAINING & EXTRACTION
# ======================================

def train_autoencoder_classifier(
        X_train, y_train, X_val, y_val,
        seq_len=LOOKBACK_SIZE,
        n_feat=NUM_FEATURES,
        latent_dim=LATENT_DIM,
        alpha=ALPHA,
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
):
    """
    Train the sequence autoencoder + classifier. Returns the trained model,
    the standalone encoder, and the training history.
    """
    model, encoder_model = build_seq_autoencoder_with_classifier(
        seq_len=seq_len,
        n_feat=n_feat,
        latent_dim=latent_dim,
        alpha=alpha,
        lr=lr
    )

    print(model.summary())

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    f1_callback = F1Metrics(X_val, y_val)

    # Force CPU usage if desired; remove `with tf.device('/CPU:0')` if you want GPU
    with tf.device('/CPU:0'):
        history = model.fit(
            x=X_train,
            y={"reconstruction": X_train, "classifier": y_train},
            validation_data=(X_val, {"reconstruction": X_val, "classifier": y_val}),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[EarlyStopping(patience=5,
                                     restore_best_weights=True,
                                     verbose=1
                                     ),
                       reduce_lr,
                       f1_callback
                       ],
            verbose=1
        )

    return model, encoder_model, history


def extract_embeddings(encoder, X_all):
    z_all = encoder.predict(X_all, batch_size=BATCH_SIZE)
    return z_all


def cluster_bad_prototypes(z_all, y_all, n_prototypes=N_BAD_PROTOTYPES):
    bad_indices = np.where(y_all == 0)[0]
    z_bad = z_all[bad_indices]
    kmeans = KMeans(n_clusters=n_prototypes, random_state=RANDOM_STATE).fit(z_bad)
    prototypes = kmeans.cluster_centers_
    return prototypes


# ======================================
#        PROTOTYPE CLASSIFICATION
# ======================================

def classify_by_prototype(window_raw, encoder, scaler, prototypes, threshold):
    # 1) Flatten & scale
    flat = window_raw.reshape(LOOKBACK_SIZE, NUM_FEATURES)
    scaled_flat = scaler.transform(flat)
    new_scaled = scaled_flat.reshape(1, LOOKBACK_SIZE, NUM_FEATURES)

    # 2) Get latent code
    z_new = encoder.predict(new_scaled)

    # 3) Compute distances to each prototype
    dists = np.linalg.norm(prototypes - z_new, axis=1)
    min_dist = float(np.min(dists))

    # 4) Classify
    label = 0 if min_dist < threshold else 1
    return label, min_dist


def scale_price_range(window, min_factor=0.9, max_factor=1.1):
    factor = np.random.uniform(min_factor, max_factor)
    window_scaled = window.copy()
    window_scaled[:, :4] *= factor  # scale open, high, low, close
    return window_scaled


def shift_price_range(window, max_shift=5.0):
    shift = np.random.uniform(-max_shift, max_shift)
    window_shifted = window.copy()
    window_shifted[:, :4] += shift  # apply shift to OHLC
    return window_shifted


def distort_volume_and_change(window, volume_scale=0.2, pct_change_scale=0.05):
    window_distorted = window.copy()
    window_distorted[:, 4] *= np.random.uniform(1 - volume_scale, 1 + volume_scale)
    window_distorted[:, 5] *= np.random.uniform(1 - pct_change_scale, 1 + pct_change_scale)
    return window_distorted


def jitter_time_series(window, sigma=0.01):
    noise = np.random.normal(loc=0.0, scale=sigma, size=window.shape)
    return window + noise


def augment_window(window):
    window = scale_price_range(window, 0.9, 1.1)
    window = shift_price_range(window, max_shift=2)
    window = distort_volume_and_change(window, 0.001, 0.002)
    window = jitter_time_series(window, sigma=0.001)
    return window


# ======================================
#         EVALUATION ON NEW FILE
# ======================================
def evaluate_with_confusion(model, scaler, X_train_scaled, json_path, threshold=0.5):
    # 1) Load raw data + true labels
    X_val_raw, y_val = load_notifications_from_jsonl(json_path)  # shape = (N_val, 19, 6)
    n_val, seq_len, n_feat = X_val_raw.shape

    # 2) Shape check
    assert seq_len == X_train_scaled.shape[1] and n_feat == X_train_scaled.shape[2], (
        f"Expected input shape {(X_train_scaled.shape[1], X_train_scaled.shape[2])}, "
        f"got {(seq_len, n_feat)}"
    )

    # 3) Flatten + scale
    flat_val = X_val_raw.reshape(n_val * seq_len, n_feat)
    flat_val_scaled = scaler.transform(flat_val)
    X_val_scaled = flat_val_scaled.reshape(n_val, seq_len, n_feat)

    # 4) Run predictions
    #    recon_preds we can ignore; cls_probs is shape (N_val, 1)
    _, cls_probs = model.predict(X_val_scaled, batch_size=1)
    cls_probs = cls_probs.reshape(-1)  # shape = (N_val,)

    # 5) Build per‐sample summary
    print("\nPer‐notification predictions:")
    print(" idx |    prob    | pred_class | true_label")
    print("-----+------------+------------+-----------")
    y_pred = np.zeros_like(cls_probs, dtype=int)
    for i, p in enumerate(cls_probs):
        y_hat = 1 if p >= threshold else 0
        y_pred[i] = y_hat
        print(f"{i:4d} | {p:8.4f}   | {y_hat:10d} | {y_val[i]:9d}")

    # 6) Confusion matrix + classification report
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
    cr = classification_report(y_val, y_pred, digits=4)

    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print("         pred=0    pred=1")
    print(f"true=0   {cm[0, 0]:8d}   {cm[0, 1]:8d}")
    print(f"true=1   {cm[1, 0]:8d}   {cm[1, 1]:8d}")

    print("\nClassification Report:\n")
    print(cr)


# ======================================
#               MAIN
# ======================================
if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")

    # ---- Print current configuration ----
    print("===== RUNNING WITH CONFIGURATION =====")
    print(f"LOOKBACK_SIZE   = {LOOKBACK_SIZE}")
    print(f"VALIDATE_SIZE   = {VALIDATE_SIZE}")
    print(f"LEARNING_RATE   = {LEARNING_RATE}")
    print(f"N_BAD_PROTOTYPES= {N_BAD_PROTOTYPES}")
    print("=======================================\n")

    # --- Load all data and do split/scale ---
    X_all, y_all = load_notifications_from_jsonl(JSON_PATH)
    X_train, X_test, y_train, y_test, scaler = time_based_split_and_scale(X_all, y_all)

    augmented_X = []
    augmented_y = []

    # 3) For each original window, call augment_window(...) K times
    for i in range(len(X_train)):
        for _ in range(TIMES_SIZE):
            augmented_X.append(augment_window(X_train[i]))
            augmented_y.append(y_train[i])

    # 4) Convert to numpy arrays and concatenate with the original training set
    augmented_X = np.array(augmented_X)  # shape = (N_train * K, LOOKBACK_SIZE, NUM_FEATURES)
    augmented_y = np.array(augmented_y)  # shape = (N_train * K,)

    X_train_aug = np.concatenate([X_train, augmented_X], axis=0)
    y_train_aug = np.concatenate([y_train, augmented_y], axis=0)

    print(f"Original train: {X_train.shape[0]} samples")
    print(f"Augmented copies: {augmented_X.shape[0]} samples")
    print(f"Combined train: {X_train_aug.shape[0]} samples")

    # 5) Now call your training routine on X_train_aug / y_train_aug
    model, encoder, history = train_autoencoder_classifier(
        X_train_aug, y_train_aug,
        X_test, y_test,
        seq_len=LOOKBACK_SIZE,
        n_feat=NUM_FEATURES,
        latent_dim=LATENT_DIM,
        alpha=ALPHA,
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # --- Prepare entire dataset for embedding & clustering ---
    N_all = X_all.shape[0]
    flat_all = X_all.reshape(N_all * LOOKBACK_SIZE, NUM_FEATURES)
    flat_all_scaled = scaler.transform(flat_all)
    X_all_scaled = flat_all_scaled.reshape(N_all, LOOKBACK_SIZE, NUM_FEATURES)

    z_all = extract_embeddings(encoder, X_all_scaled)
    prototypes = cluster_bad_prototypes(z_all, y_all, n_prototypes=N_BAD_PROTOTYPES)
    print("Computed bad‐pattern prototypes (shape):", prototypes.shape)

    # --- Evaluate on the validation JSON files ---
    # evaluate_on_new_file(model, scaler, X_train, VAL_JSON_PATH)
    evaluate_with_confusion(model, scaler, X_train, "valnotifications.json", threshold=0.5)
    #
    # 6J) Export to ONNX
    recon_name = model.outputs[0].name.split(':')[0]  # e.g. "reconstruction"
    cls_name = model.outputs[1].name.split(':')[0]  # e.g. "classifier"

    # set both as the outputs you want in the ONNX graph
    model.output_names = [recon_name, cls_name]

    seq_len, n_features = X_train.shape[1], X_train.shape[2]
    input_signature = [
        tf.TensorSpec((None, seq_len, n_features),
                      dtype=model.inputs[0].dtype,
                      name='input')
    ]

    onnx_model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature,
        opset=15
    )

    with open(ONNX_FILENAME, "wb") as f:
        f.write(onnx_model_proto.SerializeToString())
    print(f"ONNX model saved to {ONNX_FILENAME}")

    '''
    •	Your model is stable, not overfit, and achieves decent, but not stellar, predictive power (F1 ~0.73, ROC-AUC ~0.7).
	•	You’re probably limited more by input features and dataset ambiguity than by architecture or tuning.
	•	To improve: add/engineer features, tune augmentations, possibly try a slightly more expressive model if your dataset can support it.
	
	-> create better larger dataSet
	-> add additional features to the model
	-> use less ambigious train and val Samples
	'''
