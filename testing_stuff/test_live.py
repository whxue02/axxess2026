# test_live.py

import sys
import cv2
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from fall_detection.rf_based.pose_estimator import PoseEstimator
from fall_detection.rf_based.feature_engineer import FeatureEngineer, WINDOW_SIZE
from fall_detection.rf_based.fall_classifier import FallClassifier

WINDOW_STEP = WINDOW_SIZE // 5  # 50% overlap = emit every 25 frames

def run(source=0):
    estimator  = PoseEstimator()
    engineer   = FeatureEngineer()
    classifier = FallClassifier()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Could not open source: {source}")
        return

    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_pose    = mp.solutions.pose

    print("Running — press Q to quit")
    print("Status bar: WARMING UP → WATCHING → CONFIRMING... → *** FALL DETECTED ***")

    STATE_COLORS = {
        'warming_up': (128, 128, 128),  # grey
        'no_fall':    (0, 200, 0),      # green
        'confirming': (0, 165, 255),    # orange
        'fall':       (0, 0, 255),      # red
    }

    fall_time   = None
    state       = 'warming_up'
    frame_count = 0  # total frames seen
    last_emission = -WINDOW_STEP  # track when we last emitted a prediction

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):
                print("Video ended.")
            else:
                print("Webcam read failed.")
            break

        frame_count += 1

        # --- Pose estimation ---
        landmarks, _ = estimator.process_frame(frame)

        # Draw skeleton
        if landmarks is not None:
            results_for_drawing = estimator.pose.process(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            if results_for_drawing.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results_for_drawing.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255,255,255), thickness=2)
                )

        # --- Feature extraction ---
        # Always call compute() every frame so the internal window buffer
        # stays up to date — we just choose when to use the result
        flat_features = engineer.compute(landmarks)

        # --- Only run classifier every WINDOW_STEP frames (50% overlap) ---
        frames_since_last = frame_count - last_emission
        should_predict = (
            flat_features is not None and
            frames_since_last >= WINDOW_STEP
        )

        if flat_features is None:
            state = 'warming_up'
        elif should_predict:
            last_emission = frame_count
            result = classifier.predict(flat_features)

            if result == 'fall':
                state     = 'fall'
                fall_time = time.time()
            elif result == 'confirming':
                state = 'confirming'
            else:
                # Keep showing fall state for 3 seconds after detection
                if state == 'fall' and fall_time and (time.time() - fall_time) < 3.0:
                    pass
                else:
                    state = 'no_fall'
        else:
            # Between prediction steps — keep showing last state
            # but clear fall after 3 seconds
            if state == 'fall' and fall_time and (time.time() - fall_time) >= 3.0:
                state = 'no_fall'

        # --- Draw overlay ---
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

        # Show hip_y and current step in corner
        if landmarks is not None:
            hip_y = (landmarks[23, 1] + landmarks[24, 1]) / 2.0
            cv2.putText(frame, f'hip_y: {hip_y:.3f}  |  next prediction in: {WINDOW_STEP - frames_since_last} frames',
                        (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

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
    estimator.close()
    print("\nDone.")


if __name__ == '__main__':
    run(source=0)

    # To test on a video file instead:
    # run(source='path/to/your/video.mp4')