import cv2
import mediapipe as mp
print(mp.__file__)
import numpy as np 

class PoseEstimator:
    def __init__(self): # Fixed: Added underscores
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Helper to draw the landmarks on the screen
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return None, frame_bgr

        # Draw the skeleton on the frame so you can see it working
        self.mp_draw.draw_landmarks(
            frame_bgr, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        lm = results.pose_landmarks.landmark
        arr = np.array([[p.x, p.y, p.z, p.visibility] for p in lm])

        return arr, frame_bgr

    def close(self):
        self.pose.close()
