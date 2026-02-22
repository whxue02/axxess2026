# fall_detection/near_fall_detector.py

import numpy as np
from collections import deque
from enum import Enum, auto
from typing import List, Optional


# ── Thresholds ────────────────────────────────────────────────────────────────
# All position metrics are now NORMALISED BY BODY HEIGHT so they are invariant
# to how close the person is to the camera.
#
# hip_norm = (hip_y - shoulder_y) / (knee_y - shoulder_y)
#   • 0.0  = hip at shoulder level  (collapsed forward)
#   • ~1.0 = hip at knee level      (standing: hip is roughly 1x torso above knee)
#   • 1.5+ = hip well below knee    (crouching/falling)
#
# Using knees (not ankles) as the reference because feet are often out of frame.
# Thresholds are camera-distance invariant.

VELOCITY_TRIGGER    = 0.025   # normalised hip drop per frame to trigger
                               # knee-normalised scale: standing≈1.0, bigger numbers here
VELOCITY_CALM       = 0.012   # below this = calm standing (knee-normalised scale)

ACTIVITY_RATIO_SIT  = 0.55    # leg/torso ratio — below this + calm = sitting

RECOVERY_FRAMES     = 30      # frames to watch for recovery after trigger
MIN_DROP_TO_QUALIFY = 0.04    # normalised drop below baseline to count as real event
                               # knee-normalised: standing≈1.0, stumble drops to ~1.3-1.5
RECOVERY_TOLERANCE  = 0.05    # normalised: hip must return within this of baseline

BASELINE_FRAMES     = 45      # calm frames to establish standing baseline

# Sanity filter — reject frames where pose is clearly broken
MAX_ACTIVITY_RATIO  = 5.0     # ratio above this = knee landmark lost/invalid


class _State(Enum):
    IDLE      = auto()
    TRIGGERED = auto()
    RECOVERY  = auto()


class NearFallDetector:
    """
    Frame-by-frame near-fall detector using body-height-normalised hip position.

    Using normalised position makes the detector invariant to camera distance —
    the same thresholds work whether the person is 1m or 5m from the camera.

    Triple-Check algorithm:
      Step 1  Velocity spike on normalised hip position
      Step 2  Activity ratio disambiguation (sit vs fall)
      Step 3  Recovery window — did hip return toward standing baseline?
    """

    def __init__(self, debug: bool = False):
        self.debug = debug

        self._baseline_buf   = deque(maxlen=BASELINE_FRAMES)
        self._standing_norm  = None    # normalised hip position when standing

        self._prev_norm    = None
        self._velocity_buf = deque(maxlen=3)

        self._state                  = _State.IDLE
        self._recovery_frames_left   = 0
        self._lowest_norm_in_recovery = None
        self._triggered_rules        = []

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, landmarks) -> str:
        """
        landmarks : np.ndarray [33, 4] normalised [0,1] or None.
        Returns 'near_fall' | 'sitting' | 'no_event'
        """
        if landmarks is None:
            self._prev_norm       = None
            self._triggered_rules = []
            return 'no_event'

        self._triggered_rules = []

        # ── Keypoints ─────────────────────────────────────────────────────────
        hip_y      = _avg_y(landmarks, 23, 24)
        knee_y     = _avg_y(landmarks, 25, 26)
        shoulder_y = _avg_y(landmarks, 11, 12)
        # ── Body height (shoulder to knee — knees always in frame) ───────────
        body_height = abs(knee_y - shoulder_y) + 1e-6

        # ── Normalised hip position (0 = at shoulder, ~1 = at knee height) ───
        # Increases as hip drops — standing ≈ 1.0, stumbling > 1.2, fallen > 1.5
        hip_norm = (hip_y - shoulder_y) / body_height

        # ── Activity ratio ────────────────────────────────────────────────────
        leg_height   = abs(knee_y - hip_y) + 1e-6
        torso_height = abs(hip_y - shoulder_y) + 1e-6
        activity_ratio = leg_height / torso_height

        # ── Sanity check — reject broken pose frames ──────────────────────────
        if activity_ratio > MAX_ACTIVITY_RATIO or not (-0.2 <= hip_norm <= 2.5):
            if self.debug:
                print(f"[NF] REJECTED frame  hip_norm={hip_norm:.3f}  ratio={activity_ratio:.2f}")
            self._prev_norm = None
            return 'no_event'

        # ── Velocity on normalised position ───────────────────────────────────
        velocity = 0.0
        if self._prev_norm is not None:
            velocity = hip_norm - self._prev_norm
        self._prev_norm = hip_norm
        self._velocity_buf.append(velocity)
        peak_velocity = max(self._velocity_buf)

        # ── Baseline — update whenever calm, any state ────────────────────────
        if abs(peak_velocity) < VELOCITY_CALM:
            self._baseline_buf.append(hip_norm)
            if len(self._baseline_buf) >= BASELINE_FRAMES:
                self._standing_norm = float(np.mean(self._baseline_buf))

        # ── Debug ─────────────────────────────────────────────────────────────
        if self.debug:
            b   = f"{self._standing_norm:.3f}" if self._standing_norm is not None else "building"
            low = f"{self._lowest_norm_in_recovery:.3f}" if self._lowest_norm_in_recovery else "-"
            print(
                f"[NF] state={self._state.name:<10} "
                f"hip_norm={hip_norm:.3f}  vel={peak_velocity:+.4f}  "
                f"ratio={activity_ratio:.2f}  baseline={b}  "
                f"lowest={low}  recover_left={self._recovery_frames_left}"
            )

        # ── State machine ─────────────────────────────────────────────────────
        if self._state is _State.IDLE:
            # CHANGE 1: Must exceed the higher velocity threshold
            velocity_spike = peak_velocity >= VELOCITY_TRIGGER
            
            # CHANGE 2: Must be physically below the standing baseline
            # This ignores jumps/dancing where the hip is higher than normal.
            is_below_baseline = True
            if self._standing_norm is not None:
                is_below_baseline = hip_norm > (self._standing_norm + 0.01)

            if velocity_spike and is_below_baseline:
                self._triggered_rules.append('velocity_spike_below_baseline')
                self._state = _State.TRIGGERED

        elif self._state is _State.TRIGGERED:
            is_sitting = (
                activity_ratio < ACTIVITY_RATIO_SIT
                and abs(peak_velocity) < VELOCITY_TRIGGER
            )
            if is_sitting:
                self._triggered_rules.append('activity_ratio_sit')
                self._state = _State.IDLE
                return 'sitting'

            self._triggered_rules.append('entering_recovery')
            self._state                   = _State.RECOVERY
            self._recovery_frames_left    = RECOVERY_FRAMES
            self._lowest_norm_in_recovery = hip_norm

        elif self._state is _State.RECOVERY:
            self._recovery_frames_left -= 1

            # Track the deepest point of the drop
            if hip_norm > self._lowest_norm_in_recovery:
                self._lowest_norm_in_recovery = hip_norm

            # Only fire near_fall if:
            # 1. baseline is known
            # 2. hip actually dropped meaningfully (MIN_DROP_TO_QUALIFY)
            # 3. hip has now returned close to standing position
            dropped_enough = (
                self._standing_norm is not None
                and self._lowest_norm_in_recovery > self._standing_norm + MIN_DROP_TO_QUALIFY
            )
            recovered = (
                dropped_enough
                and hip_norm <= self._standing_norm + RECOVERY_TOLERANCE
            )

            if recovered:
                self._triggered_rules.append('recovery_detected')
                self._state = _State.IDLE
                self._lowest_norm_in_recovery = None
                return 'near_fall'

            if self._recovery_frames_left <= 0:
                self._state = _State.IDLE
                self._lowest_norm_in_recovery = None

        return 'no_event'

    @property
    def triggered_rules(self) -> List[str]:
        return list(self._triggered_rules)

    @property
    def state(self) -> str:
        return self._state.name

    @property
    def standing_baseline(self) -> Optional[float]:
        return self._standing_norm

    def reset(self):
        self._baseline_buf.clear()
        self._standing_norm           = None
        self._prev_norm               = None
        self._velocity_buf.clear()
        self._state                   = _State.IDLE
        self._recovery_frames_left    = 0
        self._lowest_norm_in_recovery = None
        self._triggered_rules         = []


def _avg_y(landmarks: np.ndarray, idx_a: int, idx_b: int) -> float:
    return float((landmarks[idx_a, 1] + landmarks[idx_b, 1]) / 2.0)