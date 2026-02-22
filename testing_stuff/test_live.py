# test_live.py

import sys
import cv2
import time
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fall_detection.feature_engineer import FeatureEngineer, WINDOW_SIZE
from fall_detection.fall_classifier import FallClassifier
from storage.event_logger import EventLogger

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose    = mp.solutions.pose

WINDOW_STEP = WINDOW_SIZE // 5


def run(source=0):
    pose       = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5)
    engineer   = FeatureEngineer()
    classifier = FallClassifier()
    logger     = EventLogger(fps=15)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Could not open source: {source}")
        return

    print("Running — press Q to quit")
    print("Status bar: WARMING UP → WATCHING → CONFIRMING... → *** FALL DETECTED ***")

    STATE_COLORS = {
        'warming_up': (128, 128, 128),
        'no_fall':    (0, 200, 0),
        'confirming': (0, 165, 255),
        'fall':       (0, 0, 255),
    }

    fall_time     = None
    state         = 'warming_up'
    frame_count   = 0
    last_emission = -WINDOW_STEP

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended." if isinstance(source, str) else "Webcam read failed.")
            break

        frame_count += 1

        # --- Process frame ONCE ---
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            # Extract landmarks for feature engineering
            landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                                   for lm in results.pose_landmarks.landmark])

            # Draw skeleton on real frame for live display
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

            # Draw skeleton on black canvas for saving — no background, no person
            black_frame = np.zeros_like(frame)
            mp_drawing.draw_landmarks(
                black_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
        else:
            landmarks   = None
            black_frame = np.zeros_like(frame)

        # --- Buffer black frame for saving (skeleton only, no background) ---
        saved_path = logger.add_frame(black_frame)
        if saved_path:
            print(f"\nClip saved → {saved_path}")

        # --- Feature extraction ---
        flat_features = engineer.compute(landmarks)

        # --- Only run classifier every WINDOW_STEP frames ---
        frames_since_last = frame_count - last_emission
        should_predict = (
            flat_features is not None and
            frames_since_last >= WINDOW_STEP
        )

        if flat_features is None:
            state = 'warming_up'
        elif should_predict:
            last_emission = frame_count
            result        = classifier.predict(flat_features)

            if result == 'fall':
                state     = 'fall'
                fall_time = time.time()
                logger.on_fall_detected()
            elif result == 'confirming':
                state = 'confirming'
            else:
                if state == 'fall' and fall_time and (time.time() - fall_time) < 3.0:
                    pass
                else:
                    state = 'no_fall'
        else:
            if state == 'fall' and fall_time and (time.time() - fall_time) >= 3.0:
                state = 'no_fall'

        # --- Draw status bar on live display frame ---
        color = STATE_COLORS.get(state, (128, 128, 128))
        h, w  = frame.shape[:2]

        cv2.rectangle(frame, (0, 0), (w, 60), color, -1)

        status_text = {
            'warming_up': f'WARMING UP ({frame_count}/{WINDOW_SIZE} frames)',
            'no_fall':    'WATCHING — No Fall',
            'confirming': 'CONFIRMING...',
            'fall':       '*** FALL DETECTED ***',
        }.get(state, state)

        cv2.putText(frame, status_text, (10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        if landmarks is not None:
            hip_y = (landmarks[23, 1] + landmarks[24, 1]) / 2.0
            cv2.putText(frame,
                        f'hip_y: {hip_y:.3f}  |  next prediction in: {WINDOW_STEP - frames_since_last} frames',
                        (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # --- Show real video feed with skeleton ---
        cv2.imshow('Fall Detection Test — Q to quit', frame)

        if state == 'fall':
            print(f"\r*** FALL DETECTED at frame {frame_count} ***          ", end='')
        elif state == 'confirming':
            print(f"\rFrame {frame_count}: CONFIRMING...                     ", end='')
        else:
            print(f"\rFrame {frame_count}: {state}                           ", end='')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print("\nDone.")


if __name__ == '__main__':
    run(source=0)

    # To test on a video file instead:
    # run(source='path/to/your/video.mp4')