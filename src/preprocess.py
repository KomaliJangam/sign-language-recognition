"""Preprocess collected gesture data for model training.

This module:
  1. Loads raw landmark data from CSV
  2. Normalizes/standardizes the coordinates
  3. Encodes gesture labels (e.g., "HELLO" → 0, "YES" → 1)
  4. Splits into training and testing sets
  5. Saves preprocessed data for training

Why preprocessing is necessary:
  • Normalization ensures all features are on the same scale (0-1).
    Without this, the model might focus on features with larger values.
  • Label encoding converts text labels to numbers (required by neural networks).
  • Train-test split prevents overfitting and gives honest accuracy estimates.
  • Standardization (mean=0, std=1) helps the neural network learn faster.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib


RAW_DATA_FILE = Path(__file__).parent.parent / "data" / "raw" / "gesture_data.csv"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Output files
X_TRAIN_FILE = PROCESSED_DIR / "X_train.npy"
X_TEST_FILE = PROCESSED_DIR / "X_test.npy"
y_TRAIN_FILE = PROCESSED_DIR / "y_train.npy"
y_TEST_FILE = PROCESSED_DIR / "y_test.npy"
LABEL_ENCODER_FILE = PROCESSED_DIR / "label_encoder.pkl"
SCALER_FILE = PROCESSED_DIR / "scaler.pkl"


def load_raw_data(csv_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load landmarks and labels from raw CSV file.
    
    Returns:
      X: shape (n_samples, 63)  [21 landmarks × 3 coordinates each]
      y: shape (n_samples,)     [gesture labels as strings]
    """
    
    if not csv_file.exists():
        raise FileNotFoundError(f"Raw data file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    
    if df.empty:
        raise ValueError("Raw data CSV is empty. Collect some gesture samples first.")
    
    print(f"[OK] Loaded {len(df)} samples from {csv_file.name}")
    
    # Parse landmarks
    landmarks_list = []
    for landmarks_str in df["landmarks"]:
        coords = np.array([float(x) for x in landmarks_str.split()], dtype=np.float32)
        landmarks_list.append(coords)
    
    X = np.array(landmarks_list, dtype=np.float32)  # shape: (n_samples, 63)
    y = df["gesture"].values  # shape: (n_samples,)
    
    return X, y


def normalize_landmarks(X: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """Standardize landmarks to mean=0, std=1.
    
    This helps the neural network learn faster because gradients are more stable.
    
    Args:
      X: shape (n_samples, 63)
    
    Returns:
      X_normalized: standardized landmarks
      scaler: fitted scaler object (for later use on test data)
    """
    
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    print(f"[OK] Normalized landmarks (mean=0, std=1)")
    print(f"  Before: min={X.min():.3f}, max={X.max():.3f}, mean={X.mean():.3f}")
    print(f"  After:  min={X_normalized.min():.3f}, max={X_normalized.max():.3f}, mean={X_normalized.mean():.3f}")
    
    return X_normalized, scaler


def encode_labels(y: np.ndarray) -> tuple[np.ndarray, LabelEncoder]:
    """Encode gesture labels from strings to integers.
    
    Example:
      "HELLO" → 0
      "YES" → 1
      "NO" → 2
    
    Args:
      y: gesture labels (e.g., ["HELLO", "YES", "HELLO", ...])
    
    Returns:
      y_encoded: integer labels
      encoder: fitted label encoder (for later decoding predictions)
    """
    
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    print(f"[OK] Encoded {len(encoder.classes_)} gesture classes:")
    for idx, gesture in enumerate(encoder.classes_):
        count = (y_encoded == idx).sum()
        print(f"  {gesture:15s} -> {idx}  ({count} samples)")
    
    return y_encoded, encoder


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into training and testing sets.
    
    Why we need a test set:
      • Training set: used to teach the model (80% of data)
      • Test set: used to measure real accuracy (20% of data)
      
      If we test on training data, we get inflated accuracy because the model
      has already memorized the training samples. The test set is "unseen" data
      that gives an honest measure of how well the model generalizes.
    
    Args:
      X: feature matrix
      y: labels
      test_size: fraction of data reserved for testing (e.g., 0.2 = 20%)
      random_state: seed for reproducibility
    
    Returns:
      X_train, X_test, y_train, y_test
    """
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # Ensure each class is represented in both sets
    )
    
    print(f"[OK] Split data:")
    print(f"  Training set: {len(X_train)} samples ({100*(1-test_size):.0f}%)")
    print(f"  Test set:     {len(X_test)} samples ({100*test_size:.0f}%)")
    
    return X_train, X_test, y_train, y_test


def save_preprocessed_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    encoder: LabelEncoder,
    scaler: StandardScaler,
):
    """Save preprocessed data and encoders for model training."""
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    np.save(X_TRAIN_FILE, X_train)
    np.save(X_TEST_FILE, X_test)
    np.save(y_TRAIN_FILE, y_train)
    np.save(y_TEST_FILE, y_test)
    joblib.dump(encoder, LABEL_ENCODER_FILE)
    joblib.dump(scaler, SCALER_FILE)
    
    print(f"\n[OK] Saved preprocessed data to:")
    print(f"  {X_TRAIN_FILE.name}")
    print(f"  {X_TEST_FILE.name}")
    print(f"  {y_TRAIN_FILE.name}")
    print(f"  {y_TEST_FILE.name}")
    print(f"  {LABEL_ENCODER_FILE.name}")
    print(f"  {SCALER_FILE.name}")


def preprocess():
    """Main preprocessing pipeline."""
    
    print("\n" + "=" * 60)
    print("  DATA PREPROCESSING")
    print("=" * 60 + "\n")
    
    # Step 1: Load
    print("Step 1: Loading raw data...")
    X, y = load_raw_data(RAW_DATA_FILE)
    print(f"  Shape: X={X.shape}, y={y.shape}")
    
    # Step 2: Normalize
    print("\nStep 2: Normalizing landmarks...")
    X_normalized, scaler = normalize_landmarks(X)
    
    # Step 3: Encode labels
    print("\nStep 3: Encoding gesture labels...")
    y_encoded, encoder = encode_labels(y)
    
    # Step 4: Split
    print("\nStep 4: Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(X_normalized, y_encoded)
    
    # Step 5: Save
    print("\nStep 5: Saving preprocessed data...")
    save_preprocessed_data(X_train, X_test, y_train, y_test, encoder, scaler)
    
    print("\n" + "=" * 60)
    print("[OK] PREPROCESSING COMPLETE")
    print("=" * 60)
    print("\nNext step: Run 'python src/train.py' to train the neural network model.")


if __name__ == "__main__":
    preprocess()
