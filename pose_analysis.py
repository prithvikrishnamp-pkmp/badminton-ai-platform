import cv2
import mediapipe as mp
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load pose landmark model
base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.PoseLandmarker.create_from_options(options)


def analyze_video(video_path, stframe=None):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    prev_left_ankle = None
    prev_right_ankle = None

    balance_values = []
    speed_values = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        results = detector.detect_for_video(mp_image, frame_id)
        frame_id += 1

        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]
            h, w, _ = frame.shape

            # Draw landmarks
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0,255,0), -1)

            # 🎯 BALANCE (SHOULDER ANGLE)
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]

            balance_angle = math.atan2(
                right_shoulder.y - left_shoulder.y,
                right_shoulder.x - left_shoulder.x
            )
            balance_values.append(balance_angle)

            # 🏃 FOOTWORK SPEED
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]

            lx, ly = int(left_ankle.x * w), int(left_ankle.y * h)
            rx, ry = int(right_ankle.x * w), int(right_ankle.y * h)

            if prev_left_ankle is not None:
                speed_left = math.sqrt((lx - prev_left_ankle[0])**2 + (ly - prev_left_ankle[1])**2)
                speed_right = math.sqrt((rx - prev_right_ankle[0])**2 + (ry - prev_right_ankle[1])**2)
                speed_values.append((speed_left + speed_right) / 2)

            prev_left_ankle = (lx, ly)
            prev_right_ankle = (rx, ry)

        # Show frame in Streamlit if provided
        if stframe:
            stframe.image(frame, channels="BGR")

    cap.release()

    # Return average metrics
    avg_balance = round(sum(balance_values)/len(balance_values), 3) if balance_values else 0
    avg_speed = round(sum(speed_values)/len(speed_values), 2) if speed_values else 0

    return avg_balance, avg_speed
