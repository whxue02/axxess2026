"""
split_le2i.py
-------------
Reads LE2I annotation files to find fall start/end frames,
then splits each video into fall and non-fall clips.

Expected input structure:
    le2i/
        Coffee_room_01/Coffee_room_01/
            Annotation_files/   <- .txt files
            Videos/             <- .avi files

Output structure:
    training/data/le2i/
        fall/
            Coffee_room_01_video1_fall.avi
            ...
        non_fall/
            Coffee_room_01_video1_before.avi
            Coffee_room_01_video1_after.avi
            ...

Usage:
    python split_le2i.py --input path/to/le2i --output training/data/le2i
"""

import cv2
import argparse
from pathlib import Path


# How many frames of padding to add around the fall window
# e.g. 15 = grab 15 frames before fall starts and 15 after fall ends
FALL_PADDING = 15

# Minimum clip length in frames — clips shorter than this are skipped
MIN_CLIP_FRAMES = 10


def parse_annotation(txt_path):
    """
    Reads a LE2I annotation .txt file.
    Returns (fall_start, fall_end) as integers, or (None, None) if unreadable.

    File format: first line is "START END" then per-frame data follows.
    Example first line: "48 80"
    """
    try:
        text = txt_path.read_text().strip()
        # Format: "48 80 1,1,0,0,0,0 2,1,0,0,0,0 ..."
        # First two space-separated tokens are always fall_start and fall_end
        tokens = text.split()
        return int(tokens[0]), int(tokens[1])
    except Exception as e:
        print(f"    WARNING: Could not parse {txt_path.name}: {e}")
        return None, None


def write_clip(cap, out_path, start_frame, end_frame, fps, width, height):
    """
    Extracts frames [start_frame, end_frame] from cap and writes to out_path.
    Returns True if clip was written successfully.
    """
    if end_frame - start_frame < MIN_CLIP_FRAMES:
        return False

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    out.release()
    return True


def process_scene(scene_path, output_fall, output_nonfail):
    """
    Processes one scene folder (e.g. Coffee_room_01).
    Finds matching video + annotation pairs and splits them.
    """
    # LE2I has a nested folder with the same name
    inner = scene_path / scene_path.name
    if not inner.exists():
        inner = scene_path  # some scenes aren't nested

    ann_dir = inner / 'Annotation_files'
    vid_dir = inner / 'Videos'

    # Also try alternate spellings
    if not ann_dir.exists():
        ann_dir = inner / 'Annotation_Files'
    if not vid_dir.exists():
        vid_dir = inner / 'Video'

    if not ann_dir.exists() or not vid_dir.exists():
        print(f"  Skipping {scene_path.name} — missing Annotation_files or Videos folder")
        return 0, 0

    ann_files = sorted(ann_dir.glob('*.txt'))
    print(f"  {scene_path.name}: {len(ann_files)} annotation files")

    fall_count = 0
    nonfail_count = 0

    for ann_path in ann_files:
        # Match annotation to video by number
        # Annotation: "video (1).txt" -> Video: "video (1).avi"
        video_name = ann_path.stem  # e.g. "video (1)"
        video_path = None

        for ext in ['.avi', '.mp4', '.mov']:
            candidate = vid_dir / (video_name + ext)
            if candidate.exists():
                video_path = candidate
                break

        if video_path is None:
            print(f"    WARNING: No video found for {ann_path.name}")
            continue

        fall_start, fall_end = parse_annotation(ann_path)
        if fall_start is None:
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"    WARNING: Could not open {video_path.name}")
            continue

        fps        = cap.get(cv2.CAP_PROP_FPS) or 25
        width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Add padding around fall, clamp to valid range
        clip_start = max(0, fall_start - FALL_PADDING)
        clip_end   = min(total, fall_end + FALL_PADDING)

        # Build output filename base
        scene_tag  = scene_path.name          # e.g. Coffee_room_01
        video_tag  = video_name.replace(' ', '_').replace('(', '').replace(')', '')

        # --- Fall clip ---
        fall_out = output_fall / f"{scene_tag}_{video_tag}_fall.avi"
        if write_clip(cap, fall_out, clip_start, clip_end, fps, width, height):
            fall_count += 1

        # --- Non-fall clip: frames BEFORE the fall ---
        before_end = max(0, fall_start - FALL_PADDING)
        if before_end > MIN_CLIP_FRAMES:
            before_out = output_nonfail / f"{scene_tag}_{video_tag}_before.avi"
            write_clip(cap, before_out, 0, before_end, fps, width, height)
            nonfail_count += 1

        # --- Non-fall clip: frames AFTER the fall ---
        after_start = min(total, fall_end + FALL_PADDING)
        if total - after_start > MIN_CLIP_FRAMES:
            after_out = output_nonfail / f"{scene_tag}_{video_tag}_after.avi"
            write_clip(cap, after_out, after_start, total, fps, width, height)
            nonfail_count += 1

        cap.release()

    return fall_count, nonfail_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='Path to root LE2I folder')
    parser.add_argument('--output', default='training/data/le2i', help='Output folder')
    args = parser.parse_args()

    le2i_root   = Path(args.input)
    output_root = Path(args.output)
    output_fall    = output_root / 'fall'
    output_nonfail = output_root / 'non_fall'

    output_fall.mkdir(parents=True, exist_ok=True)
    output_nonfail.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {le2i_root}")
    print(f"Output: {output_root}\n")

    total_fall    = 0
    total_nonfail = 0

    # Each subfolder is a scene (Coffee_room_01, Home_01, etc.)
    scenes = [p for p in sorted(le2i_root.iterdir()) if p.is_dir()]

    if not scenes:
        print("ERROR: No scene folders found. Check your --input path.")
        return

    for scene in scenes:
        f, nf = process_scene(scene, output_fall, output_nonfail)
        total_fall    += f
        total_nonfail += nf

    print(f"\nDone.")
    print(f"  Fall clips written:     {total_fall}")
    print(f"  Non-fall clips written: {total_nonfail}")
    print(f"  Output: {output_root}")


if __name__ == '__main__':
    main()