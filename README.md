
# Hand Gesture Volume Control System

## Project Overview
The Hand Gesture Volume Control System is a Python-based project that allows users to control their laptop volume using hand gestures through a webcam.

It detects the distance between the thumb and index finger and adjusts the system volume accordingly.

## Features
- Real-time hand detection
- Finger distance calculation
- Gesture-based volume control
- Displays volume percentage
- Visual volume bar

## Technologies Used
- Python
- OpenCV
- MediaPipe
- NumPy
- Pycaw

## How It Works
1. OpenCV captures video using the webcam.
2. MediaPipe detects hand landmarks.
3. The distance between the thumb and index finger is calculated.
4. NumPy maps the distance to a volume level.
5. Pycaw changes the laptop's system volume.

## Project Files
- volume_control.py - Main Python program
- hand_landmarker.task - Hand detection model

## Installation

Install the required libraries:

pip install opencv-python mediapipe==0.10.35 numpy pycaw comtypes

## How to Run

1. Download the project files.
2. Keep both files in the same folder.
3. Open the terminal in that folder.
4. Run:

python volume_control.py

5. Show your hand in front of the webcam.
6. Move your thumb and index finger to control the volume.

Press Q to close the camera.

## Project Status
Completed - Basic hand gesture volume control with a visual volume bar.
