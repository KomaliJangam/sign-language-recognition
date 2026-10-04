# Sign Language Recognition

A real-time **Sign Language Recognition system** that uses computer vision and deep learning to detect hand gestures through a live webcam feed and predict the corresponding sign.

The system uses **MediaPipe** to detect and extract hand landmarks, converts the landmarks into normalized feature vectors, and uses a trained **TensorFlow/Keras neural network** to classify the detected gesture. A **Streamlit web application** provides an interactive interface for real-time recognition, confidence scores, and hand-landmark visualization.

## Key Features

- Real-time hand detection using a webcam
- Sign-language gesture classification
- Hand landmark extraction using MediaPipe
- Deep learning-based gesture recognition
- Confidence-based predictions
- Real-time hand landmark visualization
- Streamlit-based interactive web interface
- Complete pipeline for data collection, preprocessing, model training, and inference

## Supported Technologies

- **Python**
- **OpenCV**
- **MediaPipe**
- **TensorFlow / Keras**
- **Streamlit**
- **NumPy**
- **Pandas**
- **scikit-learn**

## Recognized Signs

The system is designed to recognize common signs/phrases such as:

- Hello
- Yes
- No
- Thank You
- Please
- Help
- Stop
- I Love You

## Project Goal

The goal of this project is to demonstrate how **computer vision and deep learning** can be applied to real-world gesture recognition and to build an accessible system that can help bridge communication gaps for people using sign language.
