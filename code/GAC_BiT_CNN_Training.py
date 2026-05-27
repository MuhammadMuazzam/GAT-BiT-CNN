"""
=============================================================================
GAC-BiT-CNN: Model Training with 5-Fold Cross-Validation
=============================================================================
Paper: Enhancing Functional Prediction of Methyltransferases Using a Hybrid
       GAC-BiT-CNN Deep Learning Model with Protein Language Model Embeddings

Authors: Muhammad Muazzam, Farman Ali, Muhammad Asif
         Department of Computer Science, Bahria University Islamabad, Pakistan

Description:
    Trains the GAC-BiT-CNN hybrid deep learning model using 5-fold
    cross-validation on the fused feature set (4352-dim).
    Saves the trained model, weights, and architecture to disk.

Usage:
    python code/GAC_BiT_CNN_Training.py \
        --input embeddings/data_training_Fused.csv \
        --output_dir saved_model/
=============================================================================
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, matthews_corrcoef
)

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, Activation, Add, Dense, Flatten,
    Bidirectional, LSTM, LayerNormalization, Dropout,
    MultiHeadAttention, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K


# ─────────────────────────────────────────────────────────────
# CAPSULE COMPONENTS
# ─────────────────────────────────────────────────────────────

def squash(vectors, axis=-1):
    """
    Non-linear squash activation for capsule networks.
    Scales vector magnitude to range [0, 1].
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(
        s_squared_norm + K.epsilon()
    )
    return scale * vectors


class CapsuleLayer(tf.keras.layers.Layer):
    """
    Capsule Layer with dynamic routing-by-agreement.
    Implements the GAC module of the GAC-BiT-CNN architecture.

    Args:
        num_capsules (int): Number of output capsules.
        dim_capsule (int): Dimension of each capsule vector.
        routing_iters (int): Number of dynamic routing iterations.
    """

    def __init__(self, num_capsules=4, dim_capsule=8, routing_iters=3, **kwargs):
        super().__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routing_iters = routing_iters

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.num_capsules * self.dim_capsule),
            initializer='glorot_uniform',
            trainable=True,
            name='capsule_weight'
        )
        super().build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        timesteps  = tf.shape(inputs)[1]

        # Compute prediction vectors
        u_hat = tf.matmul(inputs, self.W)
        u_hat = tf.reshape(
            u_hat,
            (batch_size, timesteps, self.num_capsules, self.dim_capsule)
        )

        # Adversarial-style noise for robustness (GAN-inspired)
        noise = tf.random.normal(shape=tf.shape(u_hat), mean=0.0, stddev=0.01)
        u_hat += noise

        # Dynamic routing
        b = tf.zeros_like(u_hat[..., 0])
        for _ in range(self.routing_iters):
            c = tf.nn.softmax(b, axis=2)
            s = tf.reduce_sum(c[..., tf.newaxis] * u_hat, axis=1)
            v = squash(s)
            b += tf.reduce_sum(u_hat * tf.expand_dims(v, axis=1), axis=-1)

        return v

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_capsules': self.num_capsules,
            'dim_capsule':  self.dim_capsule,
            'routing_iters': self.routing_iters
        })
        return config


# ─────────────────────────────────────────────────────────────
# MODEL BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────

def bidirectional_transformer_block(x, num_heads=2, key_dim=32):
    """
    Bidirectional Transformer (BiT) block.
    Multi-head self-attention + feed-forward sublayer with residual connections.

    Args:
        x: Input tensor.
        num_heads (int): Number of attention heads.
        key_dim (int): Key dimension for each attention head.

    Returns:
        Output tensor after BiT processing.
    """
    # Multi-head self-attention
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x1 = Add()([x, attn])
    x1 = LayerNormalization()(x1)

    # Feed-forward sublayer
    ff = Dense(64, activation='relu')(x1)
    ff = Dense(x1.shape[-1])(ff)
    x2 = Add()([x1, ff])
    x2 = LayerNormalization()(x2)
    return x2


def residual_tcn_block(x, filters=64, kernel_size=3, dilation_rate=1):
    """
    Residual Temporal Convolutional Network (TCN) block.
    Two dilated Conv1D layers with skip connection.

    Args:
        x: Input tensor.
        filters (int): Number of convolutional filters.
        kernel_size (int): Size of convolutional kernel.
        dilation_rate (int): Dilation rate for dilated convolution.

    Returns:
        Output tensor after residual TCN processing.
    """
    shortcut = x
    x = Conv1D(filters, kernel_size, padding='same',
               dilation_rate=dilation_rate, activation='relu')(x)
    x = LayerNormalization()(x)
    x = Conv1D(filters, kernel_size, padding='same',
               dilation_rate=dilation_rate, activation='relu')(x)
    x = LayerNormalization()(x)
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x


# ─────────────────────────────────────────────────────────────
# FULL MODEL: GAC-BiT-CNN
# ─────────────────────────────────────────────────────────────

def build_gac_bitcnn(input_shape):
    """
    Build the full GAC-BiT-CNN hybrid architecture.

    Architecture:
        CNN Branch  : Multi-scale residual TCN (kernel: 3,5,7 / dilation: 1,2,3)
        BiT Branch  : Bidirectional Transformer + BiLSTM
        GAC Branch  : Capsule Network with dynamic routing
        Fusion      : Concatenate all three branches
        Classifier  : Dense(128) -> Dropout(0.3) -> Sigmoid

    Args:
        input_shape (tuple): Shape of input (feature_dim, 1).

    Returns:
        Compiled Keras Model.
    """
    inputs = Input(shape=input_shape)

    # ── CNN Branch (Multi-scale TCN) ──────────────────────────
    x1 = residual_tcn_block(inputs, filters=64, kernel_size=3, dilation_rate=1)
    x2 = residual_tcn_block(inputs, filters=64, kernel_size=5, dilation_rate=2)
    x3 = residual_tcn_block(inputs, filters=64, kernel_size=7, dilation_rate=3)
    cnn_out = Add()([x1, x2, x3])
    cnn_out_flat = Flatten()(cnn_out)

    # ── BiT Branch (Bidirectional Transformer + BiLSTM) ───────
    bit_out = bidirectional_transformer_block(cnn_out, num_heads=2, key_dim=32)
    bit_out_lstm = Bidirectional(LSTM(64, return_sequences=True))(bit_out)
    bit_out_flat = Flatten()(bit_out_lstm)

    # ── GAC Branch (Capsule Network) ──────────────────────────
    caps_out = CapsuleLayer(num_capsules=4, dim_capsule=8, routing_iters=3)(cnn_out)
    caps_out_flat = Flatten()(caps_out)

    # ── Feature Fusion ────────────────────────────────────────
    x = Concatenate()([cnn_out_flat, bit_out_flat, caps_out_flat])

    # ── Classification Head ───────────────────────────────────
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs, name="GAC_BiT_CNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ─────────────────────────────────────────────────────────────
# EVALUATION METRICS
# ─────────────────────────────────────────────────────────────

def evaluate_metrics(y_true, y_pred_prob, threshold=0.5):
    """
    Compute all seven evaluation metrics.

    Returns:
        acc, sensitivity, specificity, precision, f1, auc, mcc
    """
    y_pred = (y_pred_prob.flatten() > threshold).astype(int)

    acc         = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    precision   = precision_score(y_true, y_pred, zero_division=0)
    f1          = f1_score(y_true, y_pred)
    auc         = roc_auc_score(y_true, y_pred_prob.flatten())

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc         = matthews_corrcoef(y_true, y_pred)

    return acc, sensitivity, specificity, precision, f1, auc, mcc


# ─────────────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────

def train(file_path, output_dir="saved_model", n_folds=5, epochs=100, batch_size=32):
    """
    Full 5-fold cross-validation training pipeline.

    Args:
        file_path (str): Path to the fused feature CSV.
        output_dir (str): Directory to save the trained model.
        n_folds (int): Number of cross-validation folds.
        epochs (int): Maximum training epochs.
        batch_size (int): Training batch size.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load and preprocess data ──────────────────────────────
    print(f"\nLoading data from: {file_path}")
    data = pd.read_csv(file_path)

    # Drop non-numeric columns (e.g. Sequence_ID)
    non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"  Dropping non-numeric columns: {non_numeric}")
        data = data.drop(columns=non_numeric)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print(f"  Features: {X.shape[1]}   Samples: {X.shape[0]}")
    print(f"  Class distribution: PMT={int(y.sum())}, non-PMT={int((y==0).sum())}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    input_shape = (X.shape[1], 1)

    # ── 5-Fold Cross-Validation ───────────────────────────────
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    metrics = {k: [] for k in ['acc', 'sn', 'sp', 'pre', 'f1', 'auc', 'mcc']}

    plt.figure(figsize=(8, 6))

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\n{'─'*50}")
        print(f"  FOLD {fold_idx + 1} / {n_folds}")
        print(f"{'─'*50}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_gac_bitcnn(input_shape)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, verbose=1, min_lr=1e-6)
        ]

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=callbacks
        )

        y_prob = model.predict(X_val, verbose=0)
        acc, sn, sp, pre, f1, auc, mcc = evaluate_metrics(y_val, y_prob)

        metrics['acc'].append(acc);  metrics['sn'].append(sn)
        metrics['sp'].append(sp);   metrics['pre'].append(pre)
        metrics['f1'].append(f1);   metrics['auc'].append(auc)
        metrics['mcc'].append(mcc)

        fpr, tpr, _ = roc_curve(y_val, y_prob.flatten())
        plt.plot(fpr, tpr, label=f"Fold {fold_idx+1} (AUC={auc:.4f})")

        print(f"  Acc={acc:.4f}  Sn={sn:.4f}  Sp={sp:.4f}  "
              f"Pre={pre:.4f}  F1={f1:.4f}  AUC={auc:.4f}  MCC={mcc:.4f}")

    # ── ROC Curve Plot ────────────────────────────────────────
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Random')
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("5-Fold Cross-Validation ROC Curve — GAC-BiT-CNN", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curve_training.png")
    plt.savefig(roc_path, dpi=150)
    plt.show()
    print(f"\n  ROC curve saved: {roc_path}")

    # ── Print Final Results ───────────────────────────────────
    print("\n" + "="*55)
    print("  5-FOLD CROSS-VALIDATION RESULTS")
    print("="*55)
    print(f"  Accuracy    : {np.mean(metrics['acc']):.4f} ± {np.std(metrics['acc']):.4f}")
    print(f"  Sensitivity : {np.mean(metrics['sn']):.4f} ± {np.std(metrics['sn']):.4f}")
    print(f"  Specificity : {np.mean(metrics['sp']):.4f} ± {np.std(metrics['sp']):.4f}")
    print(f"  Precision   : {np.mean(metrics['pre']):.4f} ± {np.std(metrics['pre']):.4f}")
    print(f"  F1-score    : {np.mean(metrics['f1']):.4f} ± {np.std(metrics['f1']):.4f}")
    print(f"  AUC         : {np.mean(metrics['auc']):.4f} ± {np.std(metrics['auc']):.4f}")
    print(f"  MCC         : {np.mean(metrics['mcc']):.4f} ± {np.std(metrics['mcc']):.4f}")
    print("="*55)

    # ── Save Model ────────────────────────────────────────────
    model_path    = os.path.join(output_dir, "GAC_BiTCN.keras")
    weights_path  = os.path.join(output_dir, "GAC_BiTCN.weights.h5")
    json_path     = os.path.join(output_dir, "GAC_BiTCN.json")

    model.save(model_path)
    model.save_weights(weights_path)
    with open(json_path, "w") as f:
        f.write(model.to_json())

    print(f"\n  Model saved:")
    print(f"    Full model : {model_path}")
    print(f"    Weights    : {weights_path}")
    print(f"    Architecture: {json_path}")

    # ── Save Results to CSV ───────────────────────────────────
    results_df = pd.DataFrame(metrics)
    results_df.index = [f"Fold_{i+1}" for i in range(n_folds)]
    mean_row = results_df.mean()
    mean_row.name = "Mean"
    std_row = results_df.std()
    std_row.name = "Std"
    results_df = pd.concat([results_df, mean_row.to_frame().T, std_row.to_frame().T])
    results_path = os.path.join(output_dir, "training_cv_results.csv")
    results_df.to_csv(results_path, float_format="%.4f")
    print(f"  Results saved: {results_path}")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train GAC-BiT-CNN with 5-fold cross-validation."
    )
    parser.add_argument("--input", type=str,
                        default="embeddings/data_training_Fused.csv",
                        help="Path to fused feature CSV file")
    parser.add_argument("--output_dir", type=str, default="saved_model",
                        help="Directory to save model and results")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of cross-validation folds")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")

    args = parser.parse_args()
    train(
        file_path=args.input,
        output_dir=args.output_dir,
        n_folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
