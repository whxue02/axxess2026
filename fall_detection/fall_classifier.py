# detection/fall_classifier.py
import pickle
import numpy as np
from pathlib import Path

MODEL_PATH = Path('fall_detection/models/classifier.pkl')

class FallClassifier:
    def __init__(self, confirmation_windows=2):
        """
        confirmation_windows: how many consecutive windows must predict 'fall'
        before we declare an actual fall event.
        """
        with open(MODEL_PATH, 'rb') as f:
            saved = pickle.load(f)
        self.pipeline = saved['pipeline']
        self.threshold = saved.get('threshold', 0.5)
        self.confirmation_windows = confirmation_windows
        self._consecutive_positives = 0
        self._fall_declared = False

    def predict(self, flat_features):
        """
        flat_features: a flat list of numbers from FeatureEngineer.compute()
                       e.g. 250 numbers (50 frames x 5 features)
                       OR None if the window isn't full yet
        returns: 'fall' | 'no_fall' | 'confirming'
        """
        if flat_features is None:
            self._consecutive_positives = 0
            return 'no_fall'

        # flat_features is a plain list of numbers, pass it directly
        x = np.array([flat_features])
        proba = self.pipeline.predict_proba(x)[0, 1]

        if proba >= self.threshold:
            self._consecutive_positives += 1
        else:
            self._consecutive_positives = 0
            self._fall_declared = False

        if self._consecutive_positives >= self.confirmation_windows and not self._fall_declared:
            self._fall_declared = True
            return 'fall'
        elif self._consecutive_positives > 0:
            return 'confirming'
        return 'no_fall'

    def reset(self):
        self._consecutive_positives = 0
        self._fall_declared = False