# training/extract_keypoints.py

import cv2
import csv
import sys
from pathlib import Path
from collections import deque

sys.path.append(str(Path(__file__).resolve().parent.parent))

from detection.pose_estimator import PoseEstimator
from detection.feature_engineer import FeatureEngineer, FEATURE_COLS, WINDOW_SIZE

# ---- Configuration -------------------------------------------------------
DATASETS = {
    'le2i': Path('training/data/le2i'),
}
OUTPUT_CSV    = Path('training/data/keypoints_features.csv')
MIN_POSE_RATE = 0.80  # skip videos where pose detected in fewer than 80% of frames


# ---- Window labeling -----------------------------------------------------

def label_window(frame_labels_in_window):
    """
    A window is labeled as 'fall' if MORE THAN HALF its frames are fall frames.
    This prevents a window that merely clips the edge of a fall from being
    labeled as a fall window — the fall has to actually dominate the window.
    """
    fall_count = sum(frame_labels_in_window)
    return 1 if fall_count > (len(frame_labels_in_window) / 2) else 0


def get_frame_labels_le2i(total_frames, is_fall_video):
    """
    Le2i has no frame-level annotations, so we estimate:
      - First 30% of frames: person walking/standing normally → label 0
      - Middle 40% of frames: the fall event → label 1
      - Last 30% of frames: person lying on ground → label 1
                            (they still need help, so we keep it as fall)
    
    Non-fall videos: all frames → label 0
    """
    frame_labels = [0] * total_frames

    if is_fall_video:
        fall_start = int(total_frames * 0.30)
        fall_end   = int(total_frames * 0.70)
        for i in range(0, total_frames):  # middle + lying still = label 1
            frame_labels[i] = 1

    return frame_labels


# ---- Video processing ----------------------------------------------------

def process_video(video_path, is_fall_video, dataset_name, estimator, engineer):
    rows = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  WARNING: Could not open {video_path}")
        return rows

    # Count total frames upfront so we can build the label array
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get per-frame labels for this entire video before processing
    frame_labels = get_frame_labels_le2i(total_frames, is_fall_video)

    # Two parallel sliding windows:
    #   feature_window — built inside FeatureEngineer automatically
    #   label_window   — we track here to label each window correctly
    label_window_buffer = deque(maxlen=WINDOW_SIZE)

    frame_idx          = 0
    frames_with_pose   = 0
    windows_extracted  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get this frame's label and add to label window
        current_label = frame_labels[frame_idx] if frame_idx < len(frame_labels) else 0
        label_window_buffer.append(current_label)

        # Run pose estimation and feature extraction
        landmarks, _  = estimator.process_frame(frame)
        flat_features = engineer.compute(landmarks)

        if landmarks is not None:
            frames_with_pose += 1

        # Only emit a row when the feature window is full (returns non-None)
        if flat_features is not None and len(label_window_buffer) == WINDOW_SIZE:
            window_label = label_window(list(label_window_buffer))

            row = {
                'dataset':          dataset_name,
                'video':            video_path.name,
                'window_end_frame': frame_idx,
                'label':            window_label,
            }
            for col, val in zip(FEATURE_COLS, flat_features):
                row[col] = val

            rows.append(row)
            windows_extracted += 1

        frame_idx += 1

    cap.release()

    # ---- Pose quality filter (same as your original) ---------------------
    pose_rate = frames_with_pose / max(frame_idx, 1)
    pose_pct  = pose_rate * 100

    if pose_rate < MIN_POSE_RATE:
        print(f"    SKIPPED {video_path.name}: {frame_idx} frames, "
              f"pose only {pose_pct:.0f}% (below {MIN_POSE_RATE*100:.0f}% threshold)")
        return []

    fall_windows = sum(1 for r in rows if r['label'] == 1)
    print(f"    {video_path.name}: {frame_idx} frames → "
          f"{windows_extracted} windows "
          f"({fall_windows} fall, {windows_extracted - fall_windows} non-fall) "
          f"| pose {pose_pct:.0f}%")
    return rows


# ---- Main ----------------------------------------------------------------

def extract_all():
    estimator = PoseEstimator()
    engineer  = FeatureEngineer(window_size=WINDOW_SIZE)
    all_rows  = []

    for dataset_name, base_path in DATASETS.items():
        print(f"\nProcessing dataset: {dataset_name}")

        if not base_path.exists():
            print(f"  Skipping — directory not found: {base_path}")
            continue

        for label_str, is_fall in [('fall', True), ('non_fall', False)]:
            video_dir = base_path / label_str

            if not video_dir.exists():
                print(f"  Skipping missing folder: {video_dir}")
                continue

            video_files = (list(video_dir.glob('*.mp4')) +
                           list(video_dir.glob('*.avi')) +
                           list(video_dir.glob('*.mov')))

            print(f"  {label_str}: found {len(video_files)} videos")

            skipped = 0
            for video_path in sorted(video_files):
                engineer.reset()
                rows = process_video(video_path, is_fall, dataset_name, estimator, engineer)
                if not rows:
                    skipped += 1
                all_rows.extend(rows)

            if skipped:
                print(f"  {label_str}: skipped {skipped} low-quality videos")

    estimator.close()

    if not all_rows:
        print("\nERROR: No data extracted. Check dataset paths and video formats.")
        return

    # ---- Write CSV -------------------------------------------------------
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['dataset', 'video', 'window_end_frame', 'label'] + FEATURE_COLS

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    fall_rows     = sum(1 for r in all_rows if r['label'] == 1)
    non_fall_rows = len(all_rows) - fall_rows
    print(f"\nDone. Saved {len(all_rows)} total windows to {OUTPUT_CSV}")
    print(f"  Fall windows:     {fall_rows}")
    print(f"  Non-fall windows: {non_fall_rows}")
    print(f"  Fall percentage:  {fall_rows / len(all_rows) * 100:.1f}%")


if __name__ == '__main__':
    extract_all()