
import cv2
import mediapipe as mp
import math
import numpy as np
from pycaw.pycaw import AudioUtilities

devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=1
)

detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(img)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        h, w, _ = frame.shape

        x1, y1 = int(hand[4].x * w), int(hand[4].y * h)
        x2, y2 = int(hand[8].x * w), int(hand[8].y * h)

        distance = math.hypot(x2 - x1, y2 - y1)
        level = float(np.interp(distance, [30, 200], [0, 1]))
        volume.SetMasterVolumeLevelScalar(level, None)

        percent = int(level * 100)

        cv2.circle(frame, (x1, y1), 10, (0, 255, 0), -1)
        cv2.circle(frame, (x2, y2), 10, (0, 255, 0), -1)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

        cv2.rectangle(frame, (20, 90), (320, 125), (255, 255, 255), 2)
        bar_width = int(300 * level)
        cv2.rectangle(frame, (20, 90), (20 + bar_width, 125), (0, 255, 0), -1)

        cv2.putText(frame, f"Volume: {percent}%",
                    (20, 165), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Volume Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
detector.close()
cv2.destroyAllWindows()
