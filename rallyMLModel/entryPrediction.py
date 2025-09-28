import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
import tf2onnx
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.src.layers import MaxPooling1D, Conv1D
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

# Pip command: pip install numpy tensorflow tf2onnx scikit-learn keras

# ======================================
#           GLOBAL CONSTANTS
# ======================================

# File paths
JSON_PATH = "notifications.json"
VAL_JSON_PATH = "notifications.json"

# Window sizes
LOOKBACK_SIZE = 50

# Feature columns and dimensions
FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume']
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
THRESHOLD = 0.5
SEED = int.from_bytes(os.urandom(4), 'big')

ONNX_FILENAME = "entryPrediction.onnx"  # F
TIMES_SIZE = 6  # 6F

# -------------  CNN hyper-params  -------------
CONV_FILTERS = 16
KERNEL_SIZE = 3
POOL_SIZE = 2

WINDOW_ORDER: Tuple[str, ...] = (
    "beforeWindow",  # raw prices before
    "afterWindow",  # raw prices after
    "normalized50before",  # 50 bars normalised
    "normalized30before",  # 30 bars
    "normalized20before",  # 20 bars
    "normalized15before",  # 15 bars
)
REQ_LEN: Dict[str, int] = {
    "beforeWindow": 50,
    "afterWindow": 50,
    "normalized50before": 50,
    "normalized30before": 30,
    "normalized20before": 20,
    "normalized15before": 15,
}


def set_seeds():
    tf.config.set_visible_devices([], "GPU")
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    print(f"Seed used for this run: {SEED}")


# ======================================
#           DATA LOADING
# ======================================
def load_notifications(json_path: str | Path):
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    buffers: Dict[str, List[np.ndarray]] = {k: [] for k in WINDOW_ORDER}
    labels: List[int] = []
    skipped = 0

    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            target = obj.get("target")
            if not isinstance(target, int):
                skipped += 1
                continue

            # Validate presence + length for all windows first -----------
            if any(len(obj.get(k, [])) < REQ_LEN[k] for k in WINDOW_ORDER):
                skipped += 1
                continue

            # Transform & store -------------------------------------------
            for key in WINDOW_ORDER:
                seq = obj[key][-REQ_LEN[key]:]  # most‑recent slice
                arr = np.empty((REQ_LEN[key], len(FEATURE_COLS)), dtype=np.float32)
                for i, bar in enumerate(seq):
                    arr[i] = (
                        bar["open"],
                        bar["high"],
                        bar["low"],
                        bar["close"],
                        bar["volume"],
                    )
                buffers[key].append(arr)
            labels.append(target)

    print(f"Loaded {len(labels)} valid notifications, skipped {skipped}")

    if not labels:
        raise RuntimeError("No valid notifications – check your file.")

    stacked = [np.stack(buffers[k], axis=0) for k in WINDOW_ORDER]

    return *stacked, np.asarray(labels, dtype=np.int32)


class F1Metrics(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val, threshold):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        # Get predicted probabilities
        y_pred_prob = self.model.predict(self.X_val, verbose=0)[1].reshape(-1)
        y_pred = (y_pred_prob >= self.threshold).astype(int)

        f1 = f1_score(self.y_val, y_pred)
        print(f"\nval_f1: {f1:.4f}")

        logs = logs or {}
        logs["val_f1"] = f1


# ======================================
#         SPLIT DATA
# ======================================
def time_based_split(X, y, test_size=TEST_SPLIT, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    print(f"  Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


# ======================================
#       MODEL CONSTRUCTION
# ======================================
def build_model(
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
        X_train, y_train, X_val, y_val, seq_len=LOOKBACK_SIZE, n_feat=NUM_FEATURES,
        latent_dim=LATENT_DIM, alpha=ALPHA, lr=LEARNING_RATE, epochs=EPOCHS,
        batch_size=BATCH_SIZE
):
    """
    Train the sequence autoencoder + classifier. Returns the trained model,
    the standalone encoder, and the training history.
    """
    model, encoder_model = build_model(
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

    f1_callback = F1Metrics(X_val, y_val, THRESHOLD)

    # Force CPU usage if desired; remove `with tf.device('/CPU:0')` if you want GPU
    with tf.device('/CPU:0'):
        model.fit(
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

    return model, encoder_model


def jitter_time_series(window, sigma=0.01):
    noise = np.random.normal(loc=0.0, scale=sigma, size=window.shape)
    return window + noise


# ======================================
#         EVALUATION ON NEW FILE
# ======================================
def evaluate_with_confusion(model, threshold, data, y):
    augmented_X = []
    augmented_y = []

    for i in range(len(data)):
        for _ in range(3):
            augmented_X.append(jitter_time_series(data[i], sigma=0.004))
            augmented_y.append(y[i])

    # 4) Convert to numpy arrays and concatenate with the original training set
    augmented_X = np.array(augmented_X)  # shape = (N_train * K, LOOKBACK_SIZE, NUM_FEATURES)
    augmented_y = np.array(augmented_y)  # shape = (N_train * K,)

    x_test_aug = np.concatenate([data, augmented_X], axis=0)
    y_test_aug = np.concatenate([y, augmented_y], axis=0)

    # 4) Run predictions
    _, cls_probs = model.predict(x_test_aug, batch_size=1)
    cls_probs = cls_probs.reshape(-1)  # shape = (N_val,)

    # 5) Build per‐sample summary
    print("\nPer‐notification predictions:")
    print(" idx |    prob    | pred_class | true_label")
    print("-----+------------+------------+-----------")
    y_pred = np.zeros_like(cls_probs, dtype=int)
    for i, p in enumerate(cls_probs):
        y_hat = 1 if p >= threshold else 0
        y_pred[i] = y_hat
        print(f"{i:4d} | {p:8.4f}   | {y_hat:10d} | {y_test_aug[i]:9d}")

    # 6) Confusion matrix + classification report
    cm = confusion_matrix(y_test_aug, y_pred, labels=[0, 1])
    cr = classification_report(y_test_aug, y_pred, digits=4)

    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print("         pred=0    pred=1")
    print(f"true=0   {cm[0, 0]:8d}   {cm[0, 1]:8d}")
    print(f"true=1   {cm[1, 0]:8d}   {cm[1, 1]:8d}")

    print("\nClassification Report:\n")
    print(cr)


def save_onnx_model(model):
    recon_name = model.outputs[0].name.split(':')[0]
    cls_name = model.outputs[1].name.split(':')[0]

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


# ======================================
#               MAIN
# ======================================
if __name__ == "__main__":
    set_seeds()  # set seed for reproducibility

    # --- Load all data and do split
    before, after, n50, n30, n20, n15, y = load_notifications(JSON_PATH)
    '''
    before and after are the raw prices before and after the event (50 minutes)
    n50, n30, n20, n15 are the 50, 30, 20, 15 bars normalized before the event
    y is the target label (0 for downwards, 1 for sidewards, 2 for upwards)
    '''
    X_train, X_test, y_train, y_test = time_based_split(n50, y)

    model, encoder = train_autoencoder_classifier(
        X_train, y_train,
        X_test, y_test,
        seq_len=LOOKBACK_SIZE,
        n_feat=NUM_FEATURES,
        latent_dim=LATENT_DIM,
        alpha=ALPHA,
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # --- Evaluate on the validation JSON files ---
    before, after, n50, n30, n20, n15, y = load_notifications(VAL_JSON_PATH)
    evaluate_with_confusion(model, 0.5, n50, y)

    save_onnx_model(model)

'''
 change to a 3 state model with a percentage certainty about the increase or decrease or neutral
 
 add an autoencoder and a bottleneck to try to recreate the future. - try with longer windows as well 
 based on the future creation if it works create more synthetic samples. 

Synthetic Data Augmentation
	•	Once the autoencoder can reconstruct future sequences reliably, you can:
	•   use the encoder to give the user an idea of what might happens in the future
	•   output the certainty of the future
	•	Combine this with your primary model to see “what might happen next” and improve signal reliability.

	
	•	Once the autoencoder can reconstruct past sequences reliably, you can:
	•	Generate slightly varied versions of your sequences.
	•	Expand your training dataset for rare market conditions.
	•	This helps your CNN+LSTM learn better generalization.

Latent Space Clustering
	•	After training, you can cluster the latent representations of windows:
	•	Identify similar market regimes or patterns.
	•	Use cluster membership as a feature for your meta-model (e.g., “signals in cluster A tend to be more reliable”).
'''
