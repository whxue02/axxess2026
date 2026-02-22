# fall_detection/__init__.py
"""
fall_detection
==============
Real-time fall and near-fall detection package.

Public API
----------
DetectionPipeline   — main entry point; feed it frames, get FrameResult back
FrameResult         — dataclass returned by DetectionPipeline.process_frame()

Individual components (use directly only if you need fine-grained control):
PoseEstimator       — MediaPipe pose wrapper
FeatureEngineer     — sliding-window feature extraction for the RF model
FallClassifier      — random-forest classifier with confirmation windowing
NearFallDetector    — rules-based near-fall / sitting disambiguation

Typical usage
-------------
    from fall_detection import DetectionPipeline
    import cv2

    pipeline = DetectionPipeline()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = pipeline.process_frame(frame)
        cv2.imshow('Fall Detection', result.annotated_frame)
        if result.alert:
            print(f'ALERT  rf={result.rf_status}  nf={result.near_fall_status}')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pipeline.close()
    cap.release()
    cv2.destroyAllWindows()
"""

from .pipeline          import DetectionPipeline, FrameResult
from .pose_estimator    import PoseEstimator
from .feature_engineer  import FeatureEngineer
from .fall_classifier   import FallClassifier
from .near_fall_detector import NearFallDetector
from .event_logger import EventLogger

__all__ = [
    'DetectionPipeline',
    'FrameResult',
    'PoseEstimator',
    'FeatureEngineer',
    'FallClassifier',
    'NearFallDetector',
    'EventLogger'
]

__version__ = '0.1.0'