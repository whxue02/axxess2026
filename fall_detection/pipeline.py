# fall_detection/pipeline.py

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

from .pose_estimator    import PoseEstimator
from .feature_engineer  import FeatureEngineer
from .fall_classifier   import FallClassifier
from .near_fall_detector import NearFallDetector


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class FrameResult:
    """
    Everything the pipeline produces for a single frame.

    rf_status       : 'fall' | 'confirming' | 'no_fall'
    near_fall_status: 'near_fall' | 'sitting' | 'no_event'
    alert           : True when either classifier fires a positive result
    pose_landmarks  : raw MediaPipe landmark array [33, 4], or None
    debug_rules     : list of rule names that fired in the near-fall detector
    annotated_frame : BGR frame with skeleton + status overlays drawn on it
    """
    rf_status        : str
    near_fall_status : str
    alert            : bool
    pose_landmarks   : Optional[np.ndarray]
    debug_rules      : List[str]
    annotated_frame  : np.ndarray


# ── Colour palette ────────────────────────────────────────────────────────────
_COLOUR = {
    'fall'       : (0,   0,   255),   # red
    'confirming' : (0,   165, 255),   # orange
    'near_fall'  : (0,   255, 255),   # yellow
    'sitting'    : (255, 200,  0 ),   # teal
    'no_fall'    : (0,   200,  0 ),   # green
    'no_event'   : (0,   200,  0 ),   # green
    'no_pose'    : (180, 180, 180),   # grey
}


class DetectionPipeline:
    """
    Real-time fall and near-fall detection pipeline.

    Every frame is processed by:
      • MediaPipe  — pose estimation
      • RF path    — FeatureEngineer → FallClassifier   (window-based)
      • Rules path — NearFallDetector                   (frame-by-frame)

    Usage
    -----
    pipeline = DetectionPipeline()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        result = pipeline.process_frame(frame)
        cv2.imshow('Fall Detection', result.annotated_frame)
    pipeline.close()
    """

    def __init__(
        self,
        rf_confirmation_windows: int = 3,
        draw_skeleton: bool = True,
        show_debug_rules: bool = False,
        near_fall_debug: bool = False,   # set True to print live metrics for threshold tuning
    ):
        self._pose      = PoseEstimator()
        self._engineer  = FeatureEngineer()
        self._classifier = FallClassifier(confirmation_windows=rf_confirmation_windows)
        self._near_fall  = NearFallDetector(debug=near_fall_debug)

        self._mp_drawing      = mp.solutions.drawing_utils
        self._mp_pose         = mp.solutions.pose
        self._draw_skeleton   = draw_skeleton
        self._show_debug      = show_debug_rules

    # ── Main entry point ──────────────────────────────────────────────────────

    def process_frame(self, frame_bgr: np.ndarray) -> FrameResult:
        """
        Process a single BGR frame.
        Returns a FrameResult with all statuses and the annotated frame.
        """
        annotated = frame_bgr.copy()

        # ── 1. Pose estimation ────────────────────────────────────────────────
        landmarks, _ = self._pose.process_frame(frame_bgr)

        if landmarks is None:
            _put_text(annotated, 'No pose detected', (20, 40), _COLOUR['no_pose'])
            return FrameResult(
                rf_status        = 'no_fall',
                near_fall_status = 'no_event',
                alert            = False,
                pose_landmarks   = None,
                debug_rules      = [],
                annotated_frame  = annotated,
            )

        # ── 2. Draw skeleton ──────────────────────────────────────────────────
        if self._draw_skeleton:
            self._draw_pose(frame_bgr, annotated, landmarks)

        # ── 3. RF path ────────────────────────────────────────────────────────
        flat_features = self._engineer.compute(landmarks)
        rf_status     = self._classifier.predict(flat_features)

        # ── 4. Rules path ─────────────────────────────────────────────────────
        near_fall_status = self._near_fall.update(landmarks)
        debug_rules      = self._near_fall.triggered_rules

        # ── 5. Composite alert ────────────────────────────────────────────────
        alert = rf_status in ('fall', 'confirming') or near_fall_status == 'near_fall'

        # ── 6. Overlay labels ─────────────────────────────────────────────────
        self._draw_labels(annotated, rf_status, near_fall_status, debug_rules)

        return FrameResult(
            rf_status        = rf_status,
            near_fall_status = near_fall_status,
            alert            = alert,
            pose_landmarks   = landmarks,
            debug_rules      = debug_rules,
            annotated_frame  = annotated,
        )

    def reset(self):
        """Reset all internal state — useful between clips."""
        self._engineer.reset()
        self._classifier.reset()
        self._near_fall.reset()

    def close(self):
        """Release MediaPipe resources."""
        self._pose.close()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _draw_pose(self, original_bgr, annotated, landmarks):
        """Draw skeleton using the cached MediaPipe result — never re-processes the frame."""
        results = self._pose._last_results
        if results is not None and results.pose_landmarks:
            self._mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                self._mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self._mp_drawing.DrawingSpec(
                    color=(0, 0, 0), thickness=6, circle_radius=2),
                connection_drawing_spec=self._mp_drawing.DrawingSpec(
                    color=(0, 0, 0), thickness=6),
            )

    def _draw_labels(self, frame, rf_status, near_fall_status, debug_rules):
        h, w = frame.shape[:2]

        # RF label — top left
        rf_colour = _COLOUR.get(rf_status, (255, 255, 255))
        _put_text(frame, f'RF : {rf_status.upper()}', (20, 40), rf_colour, scale=0.8)

        # Near-fall label — below RF
        nf_colour = _COLOUR.get(near_fall_status, (255, 255, 255))
        _put_text(frame, f'NF : {near_fall_status.upper()}', (20, 75), nf_colour, scale=0.8)

        # Big alert banner — top centre
        if rf_status == 'fall':
            _banner(frame, 'FALL DETECTED', (0, 0, 255))
        elif near_fall_status == 'near_fall':
            _banner(frame, 'NEAR FALL', (0, 220, 220))
        elif rf_status == 'confirming':
            _banner(frame, 'confirming...', (0, 165, 255))

        # Optional debug rules — bottom left
        if self._show_debug and debug_rules:
            rules_str = '  '.join(debug_rules)
            _put_text(frame, f'rules: {rules_str}', (20, h - 20),
                      (200, 200, 200), scale=0.5)


# ── Drawing utilities ─────────────────────────────────────────────────────────

def _put_text(frame, text, origin, colour, scale=0.7, thickness=2):
    cv2.putText(frame, text, origin,
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, origin,
                cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness)


def _banner(frame, text, colour):
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)
    x = (w - tw) // 2
    y = 120
    cv2.rectangle(frame, (x - 10, y - th - 10), (x + tw + 10, y + 10),
                  (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, colour, 2)