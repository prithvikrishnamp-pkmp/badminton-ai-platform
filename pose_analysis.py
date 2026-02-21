import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time

base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.PoseLandmarker.create_from_options(options)


def analyze_video(video_path, stframe=None):

    cap = cv2.VideoCapture(video_path)
    prev_left_ankle = None
    prev_right_ankle = None

    balance_list, posture_list, footwork_list = [], [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        results = detector.detect_for_video(mp_image, timestamp_ms)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]
            h, w, _ = frame.shape

            # Draw skeleton
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0,255,0), -1)

            # Balance
            ls, rs = landmarks[11], landmarks[12]
            balance = abs(ls.y - rs.y)

            # Posture (spine straightness)
            lh, rh = landmarks[23], landmarks[24]
            posture = abs((ls.y + rs.y)/2 - (lh.y + rh.y)/2)

            # Footwork
            la, ra = landmarks[27], landmarks[28]
            lx, ly = int(la.x*w), int(la.y*h)
            rx, ry = int(ra.x*w), int(ra.y*h)

            if prev_left_ankle:
                speed = math.sqrt((lx-prev_left_ankle[0])**2 + (ly-prev_left_ankle[1])**2)
                footwork_list.append(speed)

            prev_left_ankle = (lx, ly)
            prev_right_ankle = (rx, ry)

            balance_list.append(balance)
            posture_list.append(posture)

        if stframe:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, use_container_width=True)
            cv2.waitKey(1)
            time.sleep(0.01)



    cap.release()

    avg_balance = sum(balance_list)/len(balance_list)
    avg_posture = sum(posture_list)/len(posture_list)
    avg_footwork = sum(footwork_list)/len(footwork_list) if footwork_list else 0

    return avg_balance, avg_posture, avg_footwork
