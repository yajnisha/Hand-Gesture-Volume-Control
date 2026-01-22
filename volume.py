import cv2
import numpy as np
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ---------------- AUDIO SETUP ----------------
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None
)
volume = cast(interface, POINTER(IAudioEndpointVolume))

min_vol, max_vol, _ = volume.GetVolumeRange()

# ---------------- MEDIAPIPE TASKS SETUP ----------------
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

hand_landmarker = HandLandmarker.create_from_options(options)

# ---------------- CAMERA SETUP ----------------
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("Press 'Q' to quit.")

# ---------------- MAIN LOOP ----------------
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_img
    )

    timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    result = hand_landmarker.detect_for_video(mp_image, timestamp)

    lm_list = []

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]
        h, w, _ = img.shape

        for idx, lm in enumerate(hand_landmarks):
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append([idx, cx, cy])
            cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1)

    if lm_list:
        x1, y1 = lm_list[4][1], lm_list[4][2]   # Thumb
        x2, y2 = lm_list[8][1], lm_list[8][2]   # Index finger

        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        length = hypot(x2 - x1, y2 - y1)

        vol = np.interp(length, [20, 200], [min_vol, max_vol])
        vol_bar = np.interp(length, [20, 200], [400, 150])
        vol_percent = np.interp(length, [20, 200], [0, 100])

        volume.SetMasterVolumeLevel(vol, None)

        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(vol_bar)), (85, 400),
                      (0, 255, 0), cv2.FILLED)

        cv2.putText(img, f'{int(vol_percent)} %',
                    (40, 430), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 3)

        if length < 40:
            cv2.circle(img, ((x1 + x2)//2, (y1 + y2)//2),
                       10, (0, 255, 0), cv2.FILLED)

    cv2.imshow("HAND GESTURE VOLUME CONTROL", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
