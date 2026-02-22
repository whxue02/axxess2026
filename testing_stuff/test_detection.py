# run_detection.py
"""
Drop-in replacement for test.py.
Runs both the RF classifier and the rules-based near-fall detector in real time.

Usage
-----
    python run_detection.py              # webcam (default)
    python run_detection.py --source 0  # explicit webcam index
    python run_detection.py --source path/to/video.mp4
"""

import argparse
import cv2
from fall_detection.rf_based import DetectionPipeline


def main(source):
    pipeline = DetectionPipeline(
        rf_confirmation_windows=2,
        draw_skeleton=True,
        show_debug_rules=True,     # shows which rules fired at the bottom of the frame
        near_fall_debug=True
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open source: {source}")
        return

    print("Running — press Q to quit\n")
    print(f"{'Frame':>6}  {'RF':>12}  {'Near-fall':>12}  {'Alert':>6}")
    print("-" * 48)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = pipeline.process_frame(frame)

        # Console log — only print when something interesting happens
        if result.rf_status != 'no_fall' or result.near_fall_status != 'no_event':
            print(f"{frame_idx:>6}  {result.rf_status:>12}  "
                  f"{result.near_fall_status:>12}  {'⚠' if result.alert else '':>6}")

        cv2.imshow('Fall Detection — Q to quit', result.annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    pipeline.close()
    cap.release()
    cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time fall + near-fall detection')
    parser.add_argument('--source', default=0,
                        help='Webcam index (int) or path to video file')
    args = parser.parse_args()

    # Convert to int if it looks like a number (webcam index)
    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass   # it's a file path — leave as string

    main(source)