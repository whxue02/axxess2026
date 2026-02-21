import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't read from webcam")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        # Draw the skeleton on the frame
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS  # draws the lines connecting joints
        )

        # Print hip positions in the terminal
        lm = results.pose_landmarks.landmark
        hip_y = (lm[23].y + lm[24].y) / 2.0
        print(f"Hip Y: {hip_y:.3f}  (0=top of frame, 1=bottom)", end='\r')

    else:
        cv2.putText(frame, "No pose detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Press Q to quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()