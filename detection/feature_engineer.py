# detection/feature_engineer.py

import numpy as np
from collections import deque

FEATURE_COLS_SINGLE = ['hip_y', 'hip_velocity', 'spine_angle', 'bbox_aspect_ratio', 'kp_variance']
WINDOW_SIZE = 50  # number of frames per window — at 15fps this is ~1.3 seconds

# Generate the flattened feature names for the window
# e.g. hip_y_0, hip_y_1, ... hip_y_19, hip_velocity_0, ...
FEATURE_COLS = [f"{feat}_f{i}" for feat in FEATURE_COLS_SINGLE for i in range(WINDOW_SIZE)]


class FeatureEngineer:
    def __init__(self, window_size=WINDOW_SIZE):
        self.window_size = window_size
        self.prev_hip_y = None
        self.y_history = deque(maxlen=10)  # for kp_variance (internal rolling window)

        # This is the sliding window — stores the last N frames of features
        self.window = deque(maxlen=window_size)

    def compute(self, landmarks):
        """
        Call this every frame. Internally builds up the sliding window.
        Returns a flat feature vector (length window_size * 5) once the window is full.
        Returns None if the window isn't full yet or landmarks is None.
        """
        if landmarks is None:
            # Gap in detection — add zeros as a placeholder so window still slides
            self.window.append([0.0] * len(FEATURE_COLS_SINGLE))
            self.prev_hip_y = None
            return None

        # ---- Compute single-frame features (same as before) ----
        hip_y = (landmarks[23, 1] + landmarks[24, 1]) / 2.0
        hip_x = (landmarks[23, 0] + landmarks[24, 0]) / 2.0

        hip_velocity = 0.0
        if self.prev_hip_y is not None:
            hip_velocity = hip_y - self.prev_hip_y
        self.prev_hip_y = hip_y

        shoulder_y = (landmarks[11, 1] + landmarks[12, 1]) / 2.0
        shoulder_x = (landmarks[11, 0] + landmarks[12, 0]) / 2.0
        dy = hip_y - shoulder_y
        dx = hip_x - shoulder_x
        spine_angle = np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6))

        all_x = landmarks[:, 0]
        all_y = landmarks[:, 1]
        bbox_w = all_x.max() - all_x.min()
        bbox_h = all_y.max() - all_y.min()
        bbox_aspect_ratio = bbox_h / (bbox_w + 1e-6)

        self.y_history.append(all_y.mean())
        kp_variance = float(np.var(self.y_history))

        frame_features = [hip_y, hip_velocity, spine_angle, bbox_aspect_ratio, kp_variance]

        # ---- Add this frame to the sliding window ----
        self.window.append(frame_features)

        # ---- Only return a prediction-ready vector when window is full ----
        if len(self.window) < self.window_size:
            return None  # still warming up

        # Flatten: [hip_y_f0, hip_y_f1, ..., kp_variance_f19]
        flat = [self.window[i][j]
                for j in range(len(FEATURE_COLS_SINGLE))
                for i in range(self.window_size)]

        return flat  # length = window_size * 5 = 100

    def reset(self):
        """Call between videos during training."""
        self.prev_hip_y = None
        self.y_history.clear()
        self.window.clear()