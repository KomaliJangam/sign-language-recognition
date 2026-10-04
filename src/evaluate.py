"""Evaluate model with detailed metrics and confusion matrix.

Metrics Explained (in simple English):

1. ACCURACY
   What: (correct predictions) / (total predictions)
   Range: 0-100%
   Interpretation:
     • 95% accuracy: model is correct 95 times out of 100
     • Good for balanced datasets (roughly equal samples per class)
     • Can be misleading if classes are imbalanced
   Example: If accuracy=90%, and we have 100 test samples,
            90 predictions are correct, 10 are wrong.

2. PRECISION (for each gesture)
   What: (correct predictions for class X) / (all predictions for class X)
   Formula: TP / (TP + FP)
   Range: 0-100%
   Interpretation:
     • How often the model is CORRECT when it says "this is gesture X"
     • High precision = low false positives
     • Use when wrong predictions are expensive
   Example: Model says "HELLO" 100 times, is correct 90 times
            Precision for HELLO = 90%

3. RECALL (for each gesture)
   What: (correct predictions for class X) / (all actual X in test set)
   Formula: TP / (TP + FN)
   Range: 0-100%
   Interpretation:
     • How often the model FINDS gesture X when it's actually present
     • High recall = low false negatives
     • Use when missing detections are expensive
   Example: There are 100 actual "HELLO" gestures, model finds 90
            Recall for HELLO = 90%

4. F1-SCORE (for each gesture)
   What: Harmonic mean of Precision and Recall
   Formula: 2 * (Precision * Recall) / (Precision + Recall)
   Range: 0-100%
   Interpretation:
     • Balanced metric between precision and recall
     • Use when both false positives and false negatives are costly
     • Best single metric for imbalanced datasets
     • F1=100%: perfect precision and recall
     • F1=0%: model predicts this class wrong most of the time

5. CONFUSION MATRIX
   What: Table showing actual vs predicted class for each sample
   Example (4 gestures):
   
              Predicted: HELLO  YES  NO  THANK_YOU
   Actual:
   HELLO              80     10   5       5        <- 80 correct HELLO
   YES                10     85   3       2        <- 85 correct YES
   NO                  5      3  90       2        <- 90 correct NO
   THANK_YOU           5      2   2      91        <- 91 correct THANK_YOU
   
   Interpretation:
     • Diagonal elements = correct predictions (should be high)
     • Off-diagonal = misclassifications (should be low)
     • Shows which gestures are confused with each other

6. WEIGHTED AVERAGE (for multi-class)
   What: Average of per-class metrics, weighted by sample count
   Interpretation:
     • Accounts for imbalanced classes
     • Most representative single number for overall performance

ROC-AUC (Receiver Operating Characteristic - Area Under Curve):
   • Not used here (designed for binary classification)
   • Measures trade-off between true positive rate and false positive rate
   • Usually used for binary problems (spam detection, disease diagnosis)

When to Use Each Metric:
   ┌─────────────────┬──────────────────────────────────┐
   │ Metric          │ Use when...                      │
   ├─────────────────┼──────────────────────────────────┤
   │ Accuracy        │ Classes are balanced              │
   │ Precision       │ False positives are expensive    │
   │ Recall          │ False negatives are expensive    │
   │ F1-Score        │ Both types of errors are costly  │
   │ Confusion Matr. │ You need to see specific errors  │
   └─────────────────┴──────────────────────────────────┘
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import tensorflow as tf
import joblib


DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_FILE = MODELS_DIR / "gesture_model.h5"
LABEL_ENCODER_FILE = DATA_DIR / "label_encoder.pkl"
PLOTS_DIR = MODELS_DIR


def load_data_and_model():
    """Load test data and trained model."""
    
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    model = tf.keras.models.load_model(MODEL_FILE)
    encoder = joblib.load(LABEL_ENCODER_FILE)
    
    print(f"[OK] Loaded test data: {X_test.shape[0]} samples")
    print(f"[OK] Loaded trained model from: {MODEL_FILE}")
    print(f"[OK] Loaded label encoder with classes: {encoder.classes_}")
    
    return X_test, y_test, model, encoder


def predict_on_test_set(model, X_test):
    """Make predictions on test set."""
    
    # Get probability outputs from model
    y_pred_proba = model.predict(X_test, verbose=0)
    
    # Get class with highest probability
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    return y_pred, y_pred_proba


def calculate_metrics(y_test, y_pred, encoder):
    """Calculate detailed metrics."""
    
    gesture_names = encoder.classes_
    
    # Overall accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Per-class metrics
    precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    # Weighted averages (accounts for class imbalance)
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=range(len(gesture_names)))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
    }


def print_metrics(metrics, gesture_names):
    """Print evaluation metrics in a readable format."""
    
    print("\n" + "=" * 80)
    print("DETAILED MODEL EVALUATION")
    print("=" * 80)
    
    # Overall accuracy
    print(f"\nOVERALL ACCURACY: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    
    # Per-class metrics
    print("\nPER-CLASS METRICS:")
    print("-" * 80)
    print(f"{'Gesture':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    
    for idx, gesture in enumerate(gesture_names):
        prec = metrics['precision'][idx]
        rec = metrics['recall'][idx]
        f1 = metrics['f1'][idx]
        print(f"{gesture:<15} {prec:>10.4f}   {rec:>10.4f}   {f1:>10.4f}   {'-':<10}")
    
    # Weighted averages
    print("-" * 80)
    print(f"{'Weighted Avg':<15} {metrics['precision_weighted']:>10.4f}   "
          f"{metrics['recall_weighted']:>10.4f}   {metrics['f1_weighted']:>10.4f}")
    print("=" * 80)


def plot_confusion_matrix(cm, gesture_names):
    """Plot confusion matrix heatmap."""
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=gesture_names,
        yticklabels=gesture_names,
        cbar_kws={'label': 'Count'},
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    cm_path = PLOTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=100)
    print(f"\n[OK] Saved confusion matrix to: {cm_path}")
    plt.close()


def plot_metrics_bars(metrics, gesture_names):
    """Plot precision, recall, F1-score as bar charts."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    x = np.arange(len(gesture_names))
    width = 0.6
    
    # Precision
    axes[0].bar(x, metrics['precision'], width, color='#1f77b4')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Precision per Gesture')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(gesture_names, rotation=45)
    axes[0].set_ylim([0, 1.0])
    axes[0].axhline(y=metrics['precision_weighted'], color='r', linestyle='--', label='Weighted Avg')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Recall
    axes[1].bar(x, metrics['recall'], width, color='#ff7f0e')
    axes[1].set_ylabel('Recall')
    axes[1].set_title('Recall per Gesture')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(gesture_names, rotation=45)
    axes[1].set_ylim([0, 1.0])
    axes[1].axhline(y=metrics['recall_weighted'], color='r', linestyle='--', label='Weighted Avg')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # F1-Score
    axes[2].bar(x, metrics['f1'], width, color='#2ca02c')
    axes[2].set_ylabel('F1-Score')
    axes[2].set_title('F1-Score per Gesture')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(gesture_names, rotation=45)
    axes[2].set_ylim([0, 1.0])
    axes[2].axhline(y=metrics['f1_weighted'], color='r', linestyle='--', label='Weighted Avg')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    metrics_path = PLOTS_DIR / "metrics_per_gesture.png"
    plt.savefig(metrics_path, dpi=100)
    print(f"[OK] Saved metrics chart to: {metrics_path}")
    plt.close()


def evaluate():
    """Main evaluation pipeline."""
    
    print("\n" + "=" * 80)
    print("  MODEL EVALUATION AND METRICS")
    print("=" * 80 + "\n")
    
    # Step 1: Load data and model
    print("Step 1: Loading test data and model...")
    X_test, y_test, model, encoder = load_data_and_model()
    gesture_names = encoder.classes_
    
    # Step 2: Make predictions
    print("\nStep 2: Making predictions on test set...")
    y_pred, y_pred_proba = predict_on_test_set(model, X_test)
    print(f"[OK] Generated predictions for {len(y_pred)} test samples")
    
    # Step 3: Calculate metrics
    print("\nStep 3: Calculating evaluation metrics...")
    metrics = calculate_metrics(y_test, y_pred, encoder)
    
    # Step 4: Print metrics
    print("\nStep 4: Printing detailed metrics...")
    print_metrics(metrics, gesture_names)
    
    # Step 5: Create visualizations
    print("\nStep 5: Creating visualizations...")
    plot_confusion_matrix(metrics['confusion_matrix'], gesture_names)
    plot_metrics_bars(metrics, gesture_names)
    
    # Step 6: Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nKey Takeaways:")
    print(f"  • Overall Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  • Average F1-Score: {metrics['f1_weighted']*100:.2f}%")
    print(f"  • Number of test samples: {len(y_test)}")
    print(f"  • Number of gesture classes: {len(gesture_names)}")
    
    if metrics['accuracy'] < 0.5:
        print(f"\nNote: Low accuracy is expected with synthetic data.")
        print(f"      Collect real gesture samples with 'python src/collect_data.py'")
        print(f"      for significant accuracy improvement.")
    
    print("\n" + "=" * 80)
    print("[OK] EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    evaluate()
