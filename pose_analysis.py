import cv2
import mediapipe as mp
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)

    prev_left_ankle = None
    prev_right_ankle = None

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                h, w, _ = frame.shape

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS)

                # Shoulder balance
                left_shoulder = landmarks[11]
                right_shoulder = landmarks[12]

                balance_angle = math.atan2(
                    right_shoulder.y - left_shoulder.y,
                    right_shoulder.x - left_shoulder.x
                )
                print("Balance Angle:", round(balance_angle, 3))

                # Footwork speed
                left_ankle = landmarks[27]
                right_ankle = landmarks[28]

                lx, ly = int(left_ankle.x * w), int(left_ankle.y * h)
                rx, ry = int(right_ankle.x * w), int(right_ankle.y * h)

                if prev_left_ankle is not None:
                    speed_left = math.sqrt((lx - prev_left_ankle[0])**2 + (ly - prev_left_ankle[1])**2)
                    speed_right = math.sqrt((rx - prev_right_ankle[0])**2 + (ry - prev_right_ankle[1])**2)
                    print("Footwork Speed:", round((speed_left + speed_right)/2, 2))

                prev_left_ankle = (lx, ly)
                prev_right_ankle = (rx, ry)

    cap.release()
