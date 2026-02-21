# detection/fall_classifier.py
import pickle
import numpy as np
from pathlib import Path
from collections import deque

MODEL_PATH = Path('detection/models/classifier.pkl')
FEATURE_COLS = ['hip_y', 'hip_velocity', 'spine_angle', 'bbox_aspect_ratio', 'kp_variance']

class FallClassifier:
    def __init__(self, confirmation_frames=8):
        """
        confirmation_frames: how many consecutive positive frames required
        before a fall is declared. At 15fps, 8 frames ≈ 0.5 seconds.
        """
        with open(MODEL_PATH, 'rb') as f:
            saved = pickle.load(f)
        self.pipeline = saved['pipeline']
        self.threshold = saved.get('threshold', 0.5)
        self.confirmation_frames = confirmation_frames
        self._consecutive_positives = 0
        self._fall_declared = False

    def predict(self, features: dict):
        """
        features: dict from FeatureEngineer.compute()
        returns: 'fall' | 'no_fall' | 'confirming'
        """
        if features is None:
            self._consecutive_positives = 0
            return 'no_fall'

        x = np.array([[features[col] for col in FEATURE_COLS]])
        proba = self.pipeline.predict_proba(x)[0, 1]

        if proba >= self.threshold:
            self._consecutive_positives += 1
        else:
            self._consecutive_positives = 0
            self._fall_declared = False

        if self._consecutive_positives >= self.confirmation_frames and not self._fall_declared:
            self._fall_declared = True
            return 'fall'
        elif self._consecutive_positives > 0:
            return 'confirming'
        return 'no_fall'

    def reset(self):
        self._consecutive_positives = 0
        self._fall_declared = False