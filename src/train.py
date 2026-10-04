"""Train a neural network model for gesture recognition.

IMPORTANT ARCHITECTURAL DECISION:

We use landmarks (21 x 3 = 63 coordinates) as input, NOT raw images.
Therefore, we use a DENSE neural network (MLP), NOT a CNN.

Why NOT CNN for landmarks?
  • CNN expects spatial grid structure (images with pixels)
  • Landmarks are just 21 ordered points with x,y,z coords
  • No "spatial neighborhood" between landmarks that CNN can exploit
  • Using CNN would waste computational power and parameters

Why Dense/MLP?
  • Optimized for vector inputs (flat 63-dimensional coordinates)
  • Each neuron learns patterns across ALL landmarks simultaneously
  • Fewer parameters = trains faster, less overfitting risk
  • Perfect for small-medium datasets

Architecture:
  Input (63 features) -> Dense layers -> Dropout -> Output (num_gestures)
  
  Dropout: randomly disables 30-50% of neurons during training.
    Why? Prevents overfitting by forcing network to learn redundant features.

Loss function: categorical_crossentropy
  Why? Standard for multi-class classification (gesture has 1 true class).

Optimizer: Adam
  Why? Adaptive learning rate. Works well for most neural networks.

Model Training:
  • Input: normalized landmarks (63-dimensional vectors)
  • Output: probability distribution over gesture classes
  • Training: minimize loss on training set
  • Validation: monitor generalization on test set
  • Save: best model weights to file

Overfitting Warning:
  If train_accuracy >> test_accuracy, the model is overfitting.
  Solutions:
    • Add more training data
    • Increase dropout
    • Reduce model size
    • Add L2 regularization
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import joblib


DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_FILE = Path(__file__).parent.parent / "models" / "gesture_model.h5"
HISTORY_FILE = Path(__file__).parent.parent / "models" / "training_history.pkl"
PLOTS_DIR = Path(__file__).parent.parent / "models"


def load_preprocessed_data():
    """Load normalized and split data from preprocessing step."""
    
    X_train = np.load(DATA_DIR / "X_train.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    
    print(f"[OK] Loaded preprocessed data:")
    print(f"  X_train shape: {X_train.shape}  (training features)")
    print(f"  X_test shape:  {X_test.shape}   (test features)")
    print(f"  y_train shape: {y_train.shape}  (training labels)")
    print(f"  y_test shape:  {y_test.shape}   (test labels)")
    
    return X_train, X_test, y_train, y_test


def build_model(input_shape: int, num_classes: int) -> models.Sequential:
    """Build a dense neural network for landmark-based gesture recognition.
    
    Architecture:
      Input: 63 normalized landmark coordinates
      Hidden layers with dropout to prevent overfitting
      Output: probability distribution over gesture classes
    
    Args:
      input_shape: number of input features (63 for 21 landmarks x 3 coords)
      num_classes: number of gesture classes
    
    Returns:
      Compiled Keras model
    """
    
    model = models.Sequential([
        # Input layer (implicit, defined by input_shape)
        layers.Input(shape=(input_shape,)),
        
        # First hidden layer: 128 neurons
        # ReLU activation: converts negative values to 0, keeps positive
        # Why ReLU? Introduces non-linearity so model can learn complex patterns
        layers.Dense(128, activation='relu', name='hidden1'),
        
        # Dropout: randomly disable 30% of neurons during training
        # Why? Prevents overfitting by forcing redundant feature learning
        layers.Dropout(0.3, name='dropout1'),
        
        # Second hidden layer: 64 neurons (smaller than first)
        # Progressive reduction helps model learn hierarchical features
        layers.Dense(64, activation='relu', name='hidden2'),
        layers.Dropout(0.3, name='dropout2'),
        
        # Third hidden layer: 32 neurons
        layers.Dense(32, activation='relu', name='hidden3'),
        layers.Dropout(0.2, name='dropout3'),
        
        # Output layer: num_classes neurons
        # Softmax activation: converts outputs to probability distribution (sum=1)
        # Each output represents probability of that gesture class
        layers.Dense(num_classes, activation='softmax', name='output'),
    ])
    
    # Compile: tell Keras how to optimize the model
    model.compile(
        loss='sparse_categorical_crossentropy',  # for integer labels
        optimizer='adam',  # adaptive learning rate optimizer
        metrics=['accuracy'],  # track accuracy during training
    )
    
    return model


def train_model(
    model: models.Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 100,
    batch_size: int = 8,
) -> dict:
    """Train the model and return training history.
    
    Args:
      model: compiled Keras model
      X_train, y_train: training data
      X_test, y_test: test data (used for validation)
      epochs: number of training iterations over full dataset
      batch_size: how many samples per gradient update
    
    Returns:
      history: dict with 'loss', 'accuracy', 'val_loss', 'val_accuracy'
    """
    
    print(f"\n[INFO] Training configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_test)}")
    
    # Early stopping: stop if validation loss doesn't improve for 20 epochs
    # Why? Prevents overfitting and saves training time
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
    )
    
    print("\n[INFO] Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1,
    )
    
    return history.history


def evaluate_model(
    model: models.Sequential,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Evaluate model on test set and print metrics."""
    
    print("\n[INFO] Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print("=" * 60)


def plot_training_history(history: dict) -> None:
    """Plot training and validation accuracy/loss."""
    
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    axes[0].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history['val_accuracy'], label='Test Accuracy', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history['val_loss'], label='Test Loss', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = PLOTS_DIR / "training_curves.png"
    plt.savefig(plot_path, dpi=100)
    print(f"\n[OK] Saved training curves to: {plot_path}")
    plt.close()


def save_model(model: models.Sequential, history: dict) -> None:
    """Save trained model and training history."""
    
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    model.save(MODEL_FILE)
    joblib.dump(history, HISTORY_FILE)
    
    print(f"\n[OK] Saved model to: {MODEL_FILE}")
    print(f"[OK] Saved training history to: {HISTORY_FILE}")


def train():
    """Main training pipeline."""
    
    print("\n" + "=" * 60)
    print("  NEURAL NETWORK TRAINING FOR GESTURE RECOGNITION")
    print("=" * 60 + "\n")
    
    # Step 1: Load data
    print("Step 1: Loading preprocessed data...")
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Determine number of gesture classes
    num_classes = len(np.unique(y_train))
    input_features = X_train.shape[1]
    
    print(f"  Number of gesture classes: {num_classes}")
    print(f"  Number of input features: {input_features}")
    
    # Step 2: Build model
    print("\nStep 2: Building dense neural network...")
    model = build_model(input_features, num_classes)
    model.summary()
    
    # Step 3: Train model
    print("\nStep 3: Training model...")
    history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Step 4: Evaluate model
    print("\nStep 4: Evaluating model...")
    evaluate_model(model, X_test, y_test)
    
    # Step 5: Save results
    print("\nStep 5: Saving model and training history...")
    save_model(model, history)
    
    # Step 6: Plot training curves
    print("\nStep 6: Plotting training curves...")
    plot_training_history(history)
    
    print("\n" + "=" * 60)
    print("[OK] TRAINING COMPLETE")
    print("=" * 60)
    print("\nNext step: Run 'python src/evaluate.py' for detailed metrics.")
    print("Then: Run 'streamlit run app.py' to start the web application.")


if __name__ == "__main__":
    train()
