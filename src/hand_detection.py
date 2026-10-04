"""MediaPipe hand detection helpers.

This module keeps webcam-frame handling separate from the user interface and
model code. MediaPipe expects RGB images, while OpenCV captures BGR images.
"""

from __future__ import annotations

from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def create_hand_detector(
    *,
    static_image_mode: bool = False,
    max_num_hands: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> mp_hands.Hands:
    """Create and configure a MediaPipe Hands detector."""

    return mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def detect_hands(
    frame: np.ndarray,
    detector: mp_hands.Hands,
) -> object:
    """Detect hands in one OpenCV BGR frame."""

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = detector.process(rgb_frame)
    rgb_frame.flags.writeable = True
    return results


def draw_hand_landmarks(
    frame: np.ndarray,
    results: object,
) -> np.ndarray:
    """Draw detected hand skeletons on a copy of the input frame."""

    output = frame.copy()
    multi_hand_landmarks = getattr(results, "multi_hand_landmarks", None)
    if not multi_hand_landmarks:
        return output

    for hand_landmarks in multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            output,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )
    return output


def extract_landmarks(
    results: object,
    *,
    hand_index: int = 0,
) -> Optional[np.ndarray]:
    """Return one hand's 21 normalized ``(x, y, z)`` landmarks.

    The returned shape is ``(21, 3)``. ``None`` means no suitable hand was
    detected. Coordinates are already normalized by MediaPipe to approximately
    the range 0 to 1 for x and y.
    """

    multi_hand_landmarks = getattr(results, "multi_hand_landmarks", None)
    if not multi_hand_landmarks or hand_index >= len(multi_hand_landmarks):
        return None

    hand = multi_hand_landmarks[hand_index]
    return np.array(
        [[landmark.x, landmark.y, landmark.z] for landmark in hand.landmark],
        dtype=np.float32,
    )
