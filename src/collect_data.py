"""Collect hand gesture dataset from webcam.

This script allows you to record hand gestures and save them as landmark
coordinates to a CSV file. You control:

  - Gesture name (e.g., "HELLO", "YES", "NO")
  - Number of samples to capture per gesture
  - Recording duration per sample

Usage:
  python src/collect_data.py

The script saves data to: data/raw/gesture_data.csv

CSV columns:
  - gesture (label)
  - landmarks (21×3 coordinates as space-separated values)
"""

from __future__ import annotations

import os
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from hand_detection import create_hand_detector, detect_hands, extract_landmarks, draw_hand_landmarks


DATA_FILE = Path(__file__).parent.parent / "data" / "raw" / "gesture_data.csv"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
DEFAULT_CAPTURE_TIME = 2  # seconds per sample


def initialize_csv():
    """Create the CSV file with headers if it doesn't exist."""
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    if not DATA_FILE.exists():
        with open(DATA_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gesture", "landmarks"])
        print(f"[OK] Created dataset file: {DATA_FILE}")
    else:
        print(f"[OK] Using existing dataset file: {DATA_FILE}")


def landmarks_to_string(landmarks: np.ndarray) -> str:
    """Convert 21×3 landmark array to space-separated string."""
    flattened = landmarks.flatten()
    return " ".join(str(x) for x in flattened)


def collect_gesture_samples(gesture_name: str, num_samples: int, detector):
    """Capture and save samples for one gesture."""
    
    gesture_name = gesture_name.upper()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    samples_collected = 0
    
    while samples_collected < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame from webcam.")
            break
        
        # Detect hand
        results = detect_hands(frame, detector)
        landmarks = extract_landmarks(results)
        
        # Draw on screen
        display_frame = draw_hand_landmarks(frame, results)
        
        # Show info
        info_text = (
            f"Gesture: {gesture_name} | "
            f"Sample {samples_collected + 1}/{num_samples}"
        )
        cv2.putText(
            display_frame,
            info_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        
        if landmarks is None:
            cv2.putText(
                display_frame,
                "No hand detected. Please show your hand.",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            instruction = "Press 'q' to skip gesture, or show your hand to camera"
        else:
            instruction = "Press SPACE to capture sample, or 'q' to skip gesture"
        
        cv2.putText(
            display_frame,
            instruction,
            (20, FRAME_HEIGHT - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        
        # Show progress bar
        bar_width = 400
        bar_height = 30
        bar_x = FRAME_WIDTH - bar_width - 20
        bar_y = 40
        
        fill_width = int(bar_width * (samples_collected / num_samples))
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), 2)
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 255, 0), -1)
        
        cv2.imshow(f"Collect: {gesture_name}", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):  # Skip this gesture
            print(f"[SKIP] Skipped gesture '{gesture_name}'")
            break
        
        if key == ord(" ") and landmarks is not None:  # Capture sample
            with open(DATA_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                landmarks_str = landmarks_to_string(landmarks)
                writer.writerow([gesture_name, landmarks_str])
            
            samples_collected += 1
            print(f"  [OK] Captured sample {samples_collected}/{num_samples} for '{gesture_name}'")
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    print("\n" + "=" * 60)
    print("  HAND GESTURE DATA COLLECTION")
    print("=" * 60)
    print("\nThis tool captures hand gestures from your webcam")
    print("and saves them as training data.\n")
    
    initialize_csv()
    detector = create_hand_detector()
    
    while True:
        print("\nOptions:")
        print("  1. Collect new gesture samples")
        print("  2. View current dataset statistics")
        print("  3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            gesture_name = input("Enter gesture name (e.g., HELLO, YES, NO): ").strip().upper()
            if not gesture_name:
                print("❌ Gesture name cannot be empty.")
                continue
            
            try:
                num_samples = int(input("Enter number of samples to capture (e.g., 30): ").strip())
                if num_samples <= 0:
                    print("❌ Number of samples must be positive.")
                    continue
            except ValueError:
                print("❌ Please enter a valid number.")
                continue
            
            print(f"\nCollecting {num_samples} samples for gesture '{gesture_name}'...")
            print("Instructions:")
            print("  • Show your hand to the camera")
            print("  • Press SPACE to capture each sample")
            print("  • Press 'q' to skip this gesture")
            print("  • Vary hand position/angle for better model training\n")
            
            collect_gesture_samples(gesture_name, num_samples, detector)
        
        elif choice == "2":
            if not DATA_FILE.exists():
                print("❌ No dataset file found yet.")
            else:
                try:
                    with open(DATA_FILE, "r") as f:
                        reader = csv.DictReader(f)
                        gesture_counts = {}
                        for row in reader:
                            gesture = row["gesture"]
                            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
                    
                    if gesture_counts:
                        print("\n[STATS] Dataset Statistics:")
                        print(f"  Total gestures: {len(gesture_counts)}")
                        print(f"  Total samples: {sum(gesture_counts.values())}")
                        print("\n  Breakdown:")
                        for gesture, count in sorted(gesture_counts.items()):
                            print(f"    • {gesture}: {count} samples")
                    else:
                        print("❌ Dataset is empty.")
                except Exception as e:
                    print(f"❌ Error reading dataset: {e}")
        
        elif choice == "3":
            print("\n[OK] Exiting...")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")
    
    detector.close()


if __name__ == "__main__":
    main()
