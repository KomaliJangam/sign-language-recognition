from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from hand_detection import create_hand_detector, detect_hands, draw_hand_landmarks, extract_landmarks
from prediction import predict_gesture

GESTURE_MESSAGES = {
    "HELLO": "You are saying hello.",
    "YES": "You are saying yes.",
    "NO": "You are saying no.",
    "THANK_YOU": "You are saying thank you.",
    "PLEASE": "You are saying please.",
    "HELP": "You are asking for help.",
    "STOP": "You are saying stop.",
    "I_LOVE_YOU": "You are saying I love you.",
}


def _format_probability_table(probabilities: dict) -> pd.DataFrame:
    rows = [{"Gesture": label, "Confidence": confidence} for label, confidence in probabilities.items()]
    return pd.DataFrame(rows).sort_values("Confidence", ascending=False).reset_index(drop=True)


def _load_camera_frame(camera_image) -> tuple[Image.Image, np.ndarray]:
    """Decode a Streamlit camera capture to a display image and RGB pixel array."""
    image = Image.open(BytesIO(camera_image.getvalue())).convert("RGB")
    return image, np.asarray(image)


def _brighten_if_needed(frame_bgr: np.ndarray) -> np.ndarray:
    """Lift very dark camera captures so MediaPipe can see hand contours."""
    brightness = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).mean()
    if brightness >= 90:
        return frame_bgr

    gamma = 0.55
    lookup = np.array(
        [((value / 255.0) ** gamma) * 255 for value in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(frame_bgr, lookup)


def _predict_from_camera_frame(frame: np.ndarray) -> dict:
    detector = create_hand_detector()
    results = detect_hands(frame, detector)
    landmarks = extract_landmarks(results)
    if landmarks is None:
        raise ValueError("No hand detected. Please show your hand clearly in the camera.")

    result = predict_gesture(landmarks.reshape(-1))
    return result


def main() -> None:
    st.set_page_config(page_title="Sign Language Recognition", page_icon="🤟", layout="wide")
    st.title("Sign Language Recognition")
    st.caption("Predict a gesture from the live camera feed.")

    camera_image = st.camera_input("Open camera")
    frame_rgb = None

    if camera_image is not None:
        try:
            preview_image, frame_rgb = _load_camera_frame(camera_image)
            st.image(preview_image, caption="Camera preview", use_container_width=True)
        except Exception as exc:
            st.error(f"Could not read camera image: {exc}")

    if st.button("Predict gesture", type="primary"):
        if frame_rgb is None:
            st.warning("Please open the camera and capture a frame first.")
            return

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        detection_frame = _brighten_if_needed(frame_bgr)

        try:
            detector = create_hand_detector(
                static_image_mode=True,
                min_detection_confidence=0.35,
            )
            try:
                results = detect_hands(detection_frame, detector)
            finally:
                detector.close()

            landmarks = extract_landmarks(results)
            if landmarks is None:
                st.error(
                    "No hand detected. Retake the photo in brighter light, keep your "
                    "whole hand inside the frame, and face your palm toward the camera."
                )
                enhanced_rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
                st.image(
                    Image.fromarray(enhanced_rgb),
                    caption="Brightness-adjusted capture",
                    use_container_width=True,
                )
                return

            result = predict_gesture(landmarks.reshape(-1))
            predicted = result["predicted_gesture"]
            confidence = result["confidence"]

            overlay = draw_hand_landmarks(detection_frame, results)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            st.image(Image.fromarray(overlay_rgb), caption="Detected hand", use_container_width=True)

            message = GESTURE_MESSAGES.get(predicted, f"Recognized gesture: {predicted}.")
            st.success(f"{message} (Detected: {predicted}, {confidence:.2%} confidence)")

            probability_table = _format_probability_table(result["probabilities"])
            st.bar_chart(probability_table.set_index("Gesture"))
            st.dataframe(probability_table, use_container_width=True)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    st.subheader("Notes")
    st.markdown(
        """
        - This app uses your webcam and MediaPipe to detect the hand.
        - The model predicts the detected gesture based on the extracted landmarks.
        - Keep the hand clearly visible and centered in the camera frame for best results.
        """
    )


if __name__ == "__main__":
    main()
