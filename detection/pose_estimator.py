import cv2     
import mediapipe as mp        
import numpy as np 

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,   # how confident it needs to be to first find a person
            min_tracking_confidence=0.5     # how confident it needs to be to keep tracking
        )

    def process_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        results = self.pose.process(rgb)
        
        # MediaPipe couldn't find a person
        if not results.pose_landmarks:
            return None, frame_bgr
        
        lm = results.pose_landmarks.landmark
        arr = np.array([[p.x, p.y, p.z, p.visibility] for p in lm])
        # arr[23] gives you left hip: [x, y, z, visibility]
        # arr[23, 0] gives you just the x coordinate of the left hip
        # arr[23, 1] gives you just the y coordinate of the left hip
        
        return arr, frame_bgr
    
    def close(self):
        self.pose.close()