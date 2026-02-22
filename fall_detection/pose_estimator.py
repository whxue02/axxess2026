# fall_detection/pose_estimator.py

import cv2
import mediapipe as mp
import numpy as np

# Landmarks we actually need — if any of these are out of range or low
# visibility, the frame is unreliable and we return None.
_KEY_LANDMARKS = [11, 12, 23, 24, 25, 26]   # shoulders, hips, knees
_MIN_VISIBILITY = 0.4


class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._last_results = None

    def process_frame(self, frame_bgr):
        """
        Returns (landmarks_array, frame_bgr) where landmarks_array is
        np.ndarray [33, 4] (x, y, z, visibility) in normalised [0,1] coords,
        or None if no pose found or landmarks are unreliable.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        self._last_results = results

        if not results.pose_landmarks:
            return None, frame_bgr

        lm  = results.pose_landmarks.landmark
        arr = np.array([[p.x, p.y, p.z, p.visibility] for p in lm])

        # ── Validate key landmarks ────────────────────────────────────────────
        # MediaPipe can extrapolate landmarks outside [0,1] when the person
        # moves near the frame edge or very fast. Reject the whole frame if
        # any key landmark is out of range or has low visibility.
        for idx in _KEY_LANDMARKS:
            x, y, _, vis = arr[idx]
            if vis < _MIN_VISIBILITY:
                return None, frame_bgr
            if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
                return None, frame_bgr

        return arr, frame_bgr

    def close(self):
        self.pose.close()