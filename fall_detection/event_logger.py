import cv2
import os
from pathlib import Path
from collections import deque
from datetime import datetime

# How many seconds to save before and after the fall
SECONDS_BEFORE = 5
SECONDS_AFTER  = 0
FPS            = 15  # approximate webcam fps

FRAMES_BEFORE  = SECONDS_BEFORE * FPS   # 75 frames
FRAMES_AFTER   = SECONDS_AFTER  * FPS   # 45 frames

SAVE_DIR = Path('storage/fall_clips')


class EventLogger:
    def __init__(self, fps=FPS):
        self.fps = fps
        self.frames_before = SECONDS_BEFORE * fps
        self.frames_after  = SECONDS_AFTER  * fps

        # Rolling buffer — automatically drops old frames when full
        # This always holds the last FRAMES_BEFORE frames
        self.frame_buffer = deque(maxlen=self.frames_before)

        self._recording      = False   # True when a fall was detected and we're capturing post-fall frames
        self._post_fall_frames = []    # frames captured after the fall
        self._frames_remaining = 0     # how many post-fall frames still to capture

        SAVE_DIR.mkdir(parents=True, exist_ok=True)

    def add_frame(self, frame):
        """
        Call this every frame with the raw webcam frame.
        Handles both buffering pre-fall frames and capturing post-fall frames.
        Returns the filepath if a clip was just saved, otherwise None.
        """
        if not self._recording:
            # Normal operation — just keep rolling buffer of recent frames
            self.frame_buffer.append(frame.copy())
            return None
        else:
            # Fall was detected — capture post-fall frames
            self._post_fall_frames.append(frame.copy())
            self._frames_remaining -= 1

            if self._frames_remaining <= 0:
                # We have enough post-fall frames — save the clip
                filepath = self._save_clip()
                self._recording = False
                self._post_fall_frames = []
                return filepath

            return None

    def on_fall_detected(self):
        """
        Call this the moment a fall is detected.
        Starts the post-fall capture using the pre-fall buffer already collected.
        """
        if self._recording:
            return  # already recording, ignore duplicate detections

        self._recording        = True
        self._frames_remaining = self.frames_after
        self._post_fall_frames = []
        print(f"Fall detected — capturing {SECONDS_AFTER}s post-fall footage...")

    def _save_clip(self):
        """
        Combines pre-fall buffer + post-fall frames and writes to an mp4 file.
        Returns the filepath of the saved clip.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath  = SAVE_DIR / f'fall_{timestamp}.mp4'

        # Combine pre-fall buffer with post-fall frames
        all_frames = list(self.frame_buffer) + self._post_fall_frames

        if not all_frames:
            print("WARNING: No frames to save")
            return None

        # Get frame dimensions from first frame
        h, w = all_frames[0].shape[:2]

        # mp4v codec — works on Windows and Mac
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(filepath), fourcc, self.fps, (w, h))

        for frame in all_frames:
            writer.write(frame)

        writer.release()

        duration = len(all_frames) / self.fps
        print(f"Clip saved: {filepath} ({duration:.1f}s, {len(all_frames)} frames)")
        return filepath

    def reset(self):
        """Call if you want to clear the buffer (e.g. between sessions)."""
        self.frame_buffer.clear()
        self._recording        = False
        self._post_fall_frames = []
        self._frames_remaining = 0