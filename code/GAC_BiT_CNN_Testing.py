"""
=============================================================================
GAC-BiT-CNN: Independent Test Set Evaluation
=============================================================================
Paper: Enhancing Functional Prediction of Methyltransferases Using a Hybrid
       GAC-BiT-CNN Deep Learning Model with Protein Language Model Embeddings

Authors: Muhammad Muazzam, Farman Ali, Muhammad Asif
         Department of Computer Science, Bahria University Islamabad, Pakistan

Description:
    Loads a saved GAC-BiT-CNN model and evaluates it on the independent
    test set. Reports Accuracy, Sensitivity, Specificity, Precision,
    F1-score, AUC, and MCC. Generates ROC curve plot.

Usage:
    python code/GAC_BiT_CNN_Testing.py \
        --test_file embeddings/data_testing_Fused.csv \
        --model_path saved_model/GAC_BiTCN.keras \
        --output_dir results/
=============================================================================
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, matthews_corrcoef
)

import tensorflow as tf
from tensorflow.keras import backend as K


# ─────────────────────────────────────────────────────────────
# CAPSULE COMPONENTS (required for model loading)
# ─────────────────────────────────────────────────────────────

def squash(vectors, axis=-1):
    """Non-linear squash activation for capsule networks."""
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(
        s_squared_norm + K.epsilon()
    )
    return scale * vectors


class CapsuleLayer(tf.keras.layers.Layer):
    """
    Capsule Layer with dynamic routing-by-agreement.
    Must be provided as a custom object when loading the saved model.
    """

    def __init__(self, num_capsules=4, dim_capsule=8, routing_iters=3, **kwargs):
        super().__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule  = dim_capsule
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

        u_hat = tf.matmul(inputs, self.W)
        u_hat = tf.reshape(
            u_hat,
            (batch_size, timesteps, self.num_capsules, self.dim_capsule)
        )

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
            'num_capsules':  self.num_capsules,
            'dim_capsule':   self.dim_capsule,
            'routing_iters': self.routing_iters
        })
        return config


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_test_data(file_path):
    """
    Load and preprocess the test CSV file.

    Args:
        file_path (str): Path to the test feature CSV.

    Returns:
        X (np.ndarray): Preprocessed features, shape (n, features, 1).
        y (np.ndarray): Labels array.
    """
    print(f"\nLoading test data: {file_path}")
    data = pd.read_csv(file_path)

    # Drop non-numeric columns
    non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"  Dropping non-numeric columns: {non_numeric}")
        data = data.drop(columns=non_numeric)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    print(f"  Samples: {X.shape[0]}   Features: {X.shape[1]}")
    print(f"  Class distribution: PMT={int(y.sum())}, non-PMT={int((y==0).sum())}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y


# ─────────────────────────────────────────────────────────────
# EVALUATION METRICS
# ─────────────────────────────────────────────────────────────

def evaluate_metrics(y_true, y_prob, threshold=0.5):
    """
    Compute all seven evaluation metrics used in the paper.

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_prob (np.ndarray): Predicted probabilities.
        threshold (float): Classification threshold.

    Returns:
        acc, sensitivity, specificity, precision, f1, auc, mcc
    """
    y_prob = y_prob.flatten()
    y_pred = (y_prob > threshold).astype(int)

    acc         = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    precision   = precision_score(y_true, y_pred, zero_division=0)
    f1          = f1_score(y_true, y_pred, zero_division=0)
    auc         = roc_auc_score(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc         = matthews_corrcoef(y_true, y_pred)

    return acc, sensitivity, specificity, precision, f1, auc, mcc


# ─────────────────────────────────────────────────────────────
# MAIN TESTING PIPELINE
# ─────────────────────────────────────────────────────────────

def test(test_file, model_path, output_dir="results"):
    """
    Evaluate the saved GAC-BiT-CNN model on the independent test set.

    Args:
        test_file (str): Path to the test feature CSV.
        model_path (str): Path to the saved .keras model.
        output_dir (str): Directory to save results and ROC plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load test data ────────────────────────────────────────
    X, y = load_test_data(test_file)

    # ── Load saved model ──────────────────────────────────────
    print(f"\nLoading model: {model_path}")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "CapsuleLayer": CapsuleLayer,
            "squash": squash
        }
    )
    model.summary()

    # ── Predict ───────────────────────────────────────────────
    print("\nRunning predictions on independent test set...")
    y_prob = model.predict(X, verbose=1).flatten()

    # ── Compute metrics ───────────────────────────────────────
    acc, sn, sp, pre, f1, auc, mcc = evaluate_metrics(y, y_prob)

    print("\n" + "="*55)
    print("  INDEPENDENT TEST SET RESULTS")
    print("="*55)
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Sensitivity : {sn:.4f}")
    print(f"  Specificity : {sp:.4f}")
    print(f"  Precision   : {pre:.4f}")
    print(f"  F1-score    : {f1:.4f}")
    print(f"  AUC         : {auc:.4f}")
    print(f"  MCC         : {mcc:.4f}")
    print("="*55)

    # ── ROC Curve ─────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f"GAC-BiT-CNN (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Random')
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve — GAC-BiT-CNN (Independent Test Set)", fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curve_testing.png")
    plt.savefig(roc_path, dpi=150)
    plt.show()
    print(f"\n  ROC curve saved: {roc_path}")

    # ── Save predictions ──────────────────────────────────────
    pred_df = pd.DataFrame({
        "True_Label":    y,
        "Predicted_Prob": y_prob,
        "Predicted_Label": (y_prob > 0.5).astype(int)
    })
    pred_path = os.path.join(output_dir, "test_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"  Predictions saved: {pred_path}")

    # ── Save summary metrics ──────────────────────────────────
    summary = pd.DataFrame({
        "Metric": ["Accuracy", "Sensitivity", "Specificity",
                   "Precision", "F1-score", "AUC", "MCC"],
        "Value":  [acc, sn, sp, pre, f1, auc, mcc]
    })
    summary_path = os.path.join(output_dir, "test_metrics_summary.csv")
    summary.to_csv(summary_path, index=False, float_format="%.4f")
    print(f"  Metrics summary saved: {summary_path}")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate GAC-BiT-CNN on the independent test dataset."
    )
    parser.add_argument("--test_file", type=str,
                        default="embeddings/data_testing_Fused.csv",
                        help="Path to test fused feature CSV")
    parser.add_argument("--model_path", type=str,
                        default="saved_model/GAC_BiTCN.keras",
                        help="Path to saved .keras model file")
    parser.add_argument("--output_dir", type=str,
                        default="results",
                        help="Directory to save results and plots")

    args = parser.parse_args()
    test(
        test_file=args.test_file,
        model_path=args.model_path,
        output_dir=args.output_dir
    )
