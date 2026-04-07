"""train.py

Loads IMU gesture data, extracts features, trains MLP, evaluates, exports weights.

Usage:
    python train.py
    python train.py --data-dir C:/path/to/hardware
    python train.py --export-header   # also generate weights.h

Model architecture (must match FPGA): 84 -> 64 -> 32 -> 8
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from preprocess import load_and_window_csvs

# ── Config ──
GESTURE_NAMES = [
    "no_gesture", "move_forward", "turn_left", "turn_right",
    "jump", "attack", "turn_180",
]
NUM_CLASSES = 7
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "data_v2")


def augment_features(X, noise_std=0.05, n_augmented=2):
    """Add Gaussian noise augmentation to training features."""
    augmented_X = [X]
    augmented_idx = list(range(len(X)))
    for _ in range(n_augmented):
        noise = np.random.normal(0, noise_std, X.shape).astype(np.float32)
        augmented_X.append(X + noise)
        augmented_idx.extend(range(len(X)))
    return np.vstack(augmented_X), augmented_idx


def build_model(input_dim, num_classes, dropout_rate=0.3):
    """Build MLP matching FPGA architecture: 84 -> 64 -> 32 -> 8"""
    inputs = keras.Input(shape=(input_dim,), name="features", dtype="float32")
    x = layers.Dense(64, activation="relu", name="dense_64", dtype="float32")(inputs)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(32, activation="relu", name="dense_32", dtype="float32")(x)
    x = layers.Dropout(dropout_rate)(x)
    logits = layers.Dense(num_classes, activation=None, name="logits", dtype="float32")(x)
    return keras.Model(inputs=inputs, outputs=logits, name="gesture_mlp")


def export_weights_npy(model, scaler, out_dir):
    """Save model weights and scaler params as .npy files."""
    os.makedirs(out_dir, exist_ok=True)

    # Scaler params
    np.save(os.path.join(out_dir, "scaler_mean.npy"), scaler.mean_.astype(np.float32))
    np.save(os.path.join(out_dir, "scaler_scale.npy"), scaler.scale_.astype(np.float32))

    # Weight files matching weights_conversion.py expectations
    dense_layers = [l for l in model.layers if isinstance(l, layers.Dense)]
    names = ["dense1", "dense2", "output"]
    for layer, name in zip(dense_layers, names):
        W, b = layer.get_weights()
        np.save(os.path.join(out_dir, f"{name}_weights.npy"), W.astype(np.float32))
        np.save(os.path.join(out_dir, f"{name}_bias.npy"), b.astype(np.float32))
        print(f"  {name}: W{W.shape}, b{b.shape}")

    # Padded weights for FPGA (expects 8 outputs)
    fpga_dir = os.path.join(out_dir, "fpga")
    os.makedirs(fpga_dir, exist_ok=True)

    # Copy scaler and hidden layers as-is
    np.save(os.path.join(fpga_dir, "scaler_mean.npy"), scaler.mean_.astype(np.float32))
    np.save(os.path.join(fpga_dir, "scaler_scale.npy"), scaler.scale_.astype(np.float32))

    for layer, name in zip(dense_layers[:-1], names[:-1]):
        W, b = layer.get_weights()
        np.save(os.path.join(fpga_dir, f"{name}_weights.npy"), W.astype(np.float32))
        np.save(os.path.join(fpga_dir, f"{name}_bias.npy"), b.astype(np.float32))

    # Pad output layer to 8 classes
    W_out, b_out = dense_layers[-1].get_weights()
    num_model_classes = W_out.shape[1]
    fpga_classes = 8

    if num_model_classes < fpga_classes:
        pad = fpga_classes - num_model_classes
        W_out = np.pad(W_out, ((0, 0), (0, pad)), constant_values=0)
        b_out = np.pad(b_out, (0, pad), constant_values=-10.0)  # large negative bias so padded classes never win
        print(f"  Padded output: {num_model_classes} -> {fpga_classes} classes (bias=-10 for dummy classes)")

    np.save(os.path.join(fpga_dir, "output_weights.npy"), W_out.astype(np.float32))
    np.save(os.path.join(fpga_dir, "output_bias.npy"), b_out.astype(np.float32))
    print(f"  output (fpga): W{W_out.shape}, b{b_out.shape}")

    print(f"Weights saved to {out_dir}")
    print(f"FPGA weights saved to {fpga_dir}")


def export_weights_header(out_dir):
    """Generate weights.h from saved .npy files (Q8.8 fixed-point)."""
    SCALE_FACTOR = 256

    def to_fixed(x):
        scaled = np.round(x * SCALE_FACTOR)
        return np.clip(scaled, -32768, 32767).astype(np.int16)

    def fmt_1d(arr, var_name):
        lines = [f"static const MLP_DTYPE {var_name}[{len(arr)}] = {{"]
        row = "    "
        for i, val in enumerate(arr):
            row += f"{int(val)}"
            if i != len(arr) - 1:
                row += ", "
            if (i + 1) % 8 == 0 and i != len(arr) - 1:
                lines.append(row)
                row = "    "
        if row.strip():
            lines.append(row)
        lines.append("};\n")
        return "\n".join(lines)

    def fmt_2d(arr, var_name):
        rows, cols = arr.shape
        lines = [f"static const MLP_DTYPE {var_name}[{rows}][{cols}] = {{"]
        for r in range(rows):
            line = "    { "
            for c in range(cols):
                line += f"{int(arr[r, c])}"
                if c != cols - 1:
                    line += ", "
            line += " }" + ("," if r != rows - 1 else "")
            lines.append(line)
        lines.append("};\n")
        return "\n".join(lines)

    d1_w = to_fixed(np.load(os.path.join(out_dir, "dense1_weights.npy")))
    d1_b = to_fixed(np.load(os.path.join(out_dir, "dense1_bias.npy")))
    d2_w = to_fixed(np.load(os.path.join(out_dir, "dense2_weights.npy")))
    d2_b = to_fixed(np.load(os.path.join(out_dir, "dense2_bias.npy")))
    o_w  = to_fixed(np.load(os.path.join(out_dir, "output_weights.npy")))
    o_b  = to_fixed(np.load(os.path.join(out_dir, "output_bias.npy")))

    h = []
    h.append("#ifndef WEIGHTS_H")
    h.append("#define WEIGHTS_H\n")
    h.append("#include <stdint.h>\n")
    h.append("#define MLP_INPUT_SIZE 84")
    h.append("#define MLP_LAYER1_SIZE 64")
    h.append("#define MLP_LAYER2_SIZE 32")
    h.append(f"#define MLP_OUTPUT_SIZE {NUM_CLASSES}\n")
    h.append("typedef int16_t MLP_DTYPE;")
    h.append("#define MLP_SCALE_FACTOR 256")
    h.append("#define MLP_SCALE_SHIFT 8\n")
    h.append("#define FLOAT_TO_FIXED(x) ((int16_t)((x) * MLP_SCALE_FACTOR))")
    h.append("#define FIXED_TO_FLOAT(x) (((float)(x)) / MLP_SCALE_FACTOR)\n")
    h.append(fmt_2d(d1_w, "mlp_dense_64_weights"))
    h.append(fmt_1d(d1_b, "mlp_dense_64_bias"))
    h.append(fmt_2d(d2_w, "mlp_dense_32_weights"))
    h.append(fmt_1d(d2_b, "mlp_dense_32_bias"))
    h.append(fmt_2d(o_w, "mlp_output_8_weights"))
    h.append(fmt_1d(o_b, "mlp_output_8_bias"))
    h.append("#endif // WEIGHTS_H")

    header_path = os.path.join(out_dir, "weights.h")
    with open(header_path, "w") as f:
        f.write("\n".join(h))
    print(f"Header saved to {header_path}")


def main():
    parser = argparse.ArgumentParser(description="Train gesture MLP")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--step", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--augment", type=int, default=2, help="Number of noise-augmented copies")
    parser.add_argument("--export-header", action="store_true", help="Also export weights.h")
    parser.add_argument("--out-dir", default="weights_out_v2")
    args = parser.parse_args()

    # ── Load and preprocess ──
    df = load_and_window_csvs(args.data_dir, args.window_size, args.step)
    if len(df) == 0:
        print("No data found. Check your data directory.")
        sys.exit(1)

    # Separate features and labels
    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)

    # Map labels to contiguous 0..K-1
    unique_labels = np.sort(np.unique(y))
    raw_to_idx = {raw: i for i, raw in enumerate(unique_labels)}
    idx_to_raw = {i: raw for raw, i in raw_to_idx.items()}
    y_mapped = np.array([raw_to_idx[v] for v in y], dtype=np.int64)
    num_classes = len(unique_labels)

    print(f"\nLabel mapping: {raw_to_idx}")
    print(f"Num classes: {num_classes}")
    print(f"Feature shape: {X.shape}")

    # ── Train/test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_mapped, test_size=0.2, stratify=y_mapped, random_state=42
    )

    # ── Scale ──
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── Augment training data ──
    if args.augment > 0:
        X_train_aug, aug_idx = augment_features(X_train_s, noise_std=0.05, n_augmented=args.augment)
        y_train_aug = y_train[aug_idx]
        print(f"Augmented: {X_train_s.shape[0]} -> {X_train_aug.shape[0]} samples")
    else:
        X_train_aug, y_train_aug = X_train_s, y_train

    # ── Class weights ──
    cw = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train_aug)
    class_weights = {i: w for i, w in enumerate(cw)}
    print(f"Class weights: {class_weights}")

    # ── Build and train ──
    model = build_model(X_train_s.shape[1], num_classes)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_acc", mode="max", patience=100, restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train_aug, y_train_aug,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ──
    test_logits = model.predict(X_test_s, verbose=0)
    test_pred = np.argmax(test_logits, axis=1)
    test_acc = np.mean(test_pred == y_test)

    print(f"\n{'='*60}")
    print(f"TEST ACCURACY: {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"{'='*60}")

    # Map back to original gesture names for display
    display_labels = [GESTURE_NAMES[idx_to_raw[i]] if idx_to_raw[i] < len(GESTURE_NAMES)
                      else f"gesture_{idx_to_raw[i]}" for i in range(num_classes)]

    cm = confusion_matrix(y_test, test_pred, labels=range(num_classes))
    print("\nConfusion Matrix:")
    # Header
    print(f"{'':>12s}", end="")
    for name in display_labels:
        print(f"{name[:8]:>9s}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{display_labels[i]:>12s}", end="")
        for val in row:
            print(f"{val:9d}", end="")
        print()

    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, target_names=display_labels, digits=4))

    # ── Export ──
    export_weights_npy(model, scaler, args.out_dir)

    if args.export_header:
        export_weights_header(args.out_dir)

    print(f"\nDone. To test live: python live_test.py --weights-dir {args.out_dir}")


if __name__ == "__main__":
    main()
