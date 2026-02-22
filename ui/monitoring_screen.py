"""
ui/monitoring_screen.py

Live monitoring screen for the fall detection system.

Layout
------
  ┌──────────────────────────────────────────────────┐
  │  STATUS BADGE  (large, colour-coded)             │
  ├──────────────────────────────────────────────────┤
  │                                                  │
  │         SKELETON CANVAS  (centre)                │
  │                                                  │
  ├──────────────────────────────────────────────────┤
  │  Start / Stop button        RF · NF detail row   │
  └──────────────────────────────────────────────────┘

  POST-ALERT overlay (shown on confirmed fall after assessment):
  ┌──────────────────────────────────────────────────┐
  │  ⚠  FALL DETECTED                               │
  │  Assessment result + alert outcomes              │
  │  [Dismiss — return to monitoring]                │
  └──────────────────────────────────────────────────┘

Integration
-----------
  • DetectionPipeline.process_frame() runs on a background thread and
    pushes FrameResult objects to a queue.
  • The Tkinter main thread polls the queue every POLL_MS milliseconds via
    self.after() and updates the UI — safe because only the main thread
    touches Tkinter widgets.
  • When rf_status == 'fall' the assessment pipeline runs on the MAIN thread
    (pyttsx3 + sounddevice require macOS main thread).  Monitoring is paused
    while the assessment runs.
  • Falls and near-falls are logged to app.event_log via app.log_event().
"""

from __future__ import annotations

import queue
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageTk

import tkinter as tk

if TYPE_CHECKING:
    from ui.app import App

from ui.app import COLORS, FONTS, PADDING
from fall_detection import DetectionPipeline, FrameResult
from fall_detection.event_logger import EventLogger
from response import run_assessment
from response.pipeline import AssessmentResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POLL_MS       = 30    # how often the main thread polls the frame queue (~33 fps)
CANVAS_W      = 640   # skeleton canvas width  (pixels)
CANVAS_H      = 480   # skeleton canvas height (pixels)

# Status badge colours — map rf_status / near_fall_status to UI theme
_STATUS_STYLE = {
    "monitoring": {"bg": COLORS["success"],  "fg": "#FFFFFF", "label": "● MONITORING"},
    "near_fall":  {"bg": COLORS["warning"],  "fg": "#FFFFFF", "label": "⚠ NEAR FALL DETECTED"},
    "confirming": {"bg": COLORS["warning"],  "fg": "#FFFFFF", "label": "◌ CONFIRMING FALL…"},
    "fall":       {"bg": COLORS["danger"],   "fg": "#FFFFFF", "label": "⚠ FALL DETECTED"},
    "assessment": {"bg": COLORS["accent"],   "fg": "#FFFFFF", "label": "◉ RUNNING ASSESSMENT"},
    "stopped":    {"bg": COLORS["surface_raised"], "fg": COLORS["text_secondary"], "label": "○ MONITORING STOPPED"},
    "no_pose":    {"bg": COLORS["surface_raised"], "fg": COLORS["text_secondary"], "label": "○ NO POSE DETECTED"},
}


class MonitoringScreen(tk.Frame):
    """
    Live skeleton monitoring screen.

    The camera capture + fall detection run on a daemon thread.
    All UI updates happen on the main thread via a thread-safe queue.
    """

    def __init__(self, parent: tk.Widget, app: "App"):
        super().__init__(parent, bg=COLORS["bg"])
        self._app = app

        # State
        self._running      = False
        self._in_assessment = False
        self._cap          = None
        self._pipeline     = None
        self._thread       = None
        self._frame_queue: queue.Queue[FrameResult | None] = queue.Queue(maxsize=2)
        # Clip paths produced by the capture thread — drained safely on main thread
        self._clip_queue: queue.Queue[str] = queue.Queue()
        self._last_fall_time: float = 0.0   # debounce — prevent re-triggering
        self._fall_cooldown = 30.0           # seconds before a new fall can trigger

        # Overlay state
        self._overlay_visible = False
        self._last_result: AssessmentResult | None = None

        # Clip recorder — instantiated fresh each monitoring session
        self._event_logger: EventLogger | None = None
        # Maps fall log entry index → entry dict so we can patch clip_path in later
        self._pending_clip_entry: dict | None = None

        self._build()

    # -----------------------------------------------------------------------
    # Build UI
    # -----------------------------------------------------------------------

    def _build(self) -> None:

        # ── Status badge ───────────────────────────────────────────────────
        self._badge = tk.Label(
            self,
            text=_STATUS_STYLE["stopped"]["label"],
            bg=_STATUS_STYLE["stopped"]["bg"],
            fg=_STATUS_STYLE["stopped"]["fg"],
            font=(FONTS["subheading"][0], FONTS["subheading"][1], "bold"),
            anchor="center",
            pady=18,
        )
        self._badge.pack(fill=tk.X)

        # ── Centre content area ────────────────────────────────────────────
        centre = tk.Frame(self, bg=COLORS["bg"])
        centre.pack(fill=tk.BOTH, expand=True, padx=32, pady=(16, 0))

        # Skeleton canvas
        canvas_frame = tk.Frame(
            centre,
            bg=COLORS["text_primary"],          # black border around canvas
            highlightbackground=COLORS["border"],
            highlightthickness=2,
        )
        canvas_frame.pack(expand=True)

        self._canvas = tk.Canvas(
            canvas_frame,
            width=CANVAS_W,
            height=CANVAS_H,
            bg="#111111",
            highlightthickness=0,
        )
        self._canvas.pack()
        self._draw_idle_canvas()

        # ── Bottom control row ─────────────────────────────────────────────
        controls = tk.Frame(self, bg=COLORS["bg"])
        controls.pack(fill=tk.X, padx=32, pady=16)

        # Start / Stop button
        self._start_btn = tk.Label(
            controls,
            text="▶  Start Monitoring",
            bg=COLORS["success"],
            fg="#FFFFFF",
            font=FONTS["button"],
            padx=28,
            pady=14,
            cursor="hand2",
            relief=tk.FLAT,
        )
        self._start_btn.pack(side=tk.LEFT)
        self._start_btn.bind("<Button-1>", lambda e: self._toggle_monitoring())
        self._start_btn.bind("<Enter>", lambda e: self._start_btn.configure(
            bg=COLORS["danger_hover"] if self._running else "#005519"))
        self._start_btn.bind("<Leave>", lambda e: self._start_btn.configure(
            bg=COLORS["danger"] if self._running else COLORS["success"]))

        # Detail labels — RF and NF status
        detail_frame = tk.Frame(controls, bg=COLORS["bg"])
        detail_frame.pack(side=tk.RIGHT)

        tk.Label(
            detail_frame,
            text="RF Classifier",
            bg=COLORS["bg"],
            fg=COLORS["text_secondary"],
            font=FONTS["small"],
            anchor="e",
        ).grid(row=0, column=0, sticky="e", padx=(0, 8))

        self._rf_label = tk.Label(
            detail_frame,
            text="—",
            bg=COLORS["bg"],
            fg=COLORS["text_primary"],
            font=(FONTS["body"][0], FONTS["body"][1], "bold"),
            anchor="w",
            width=14,
        )
        self._rf_label.grid(row=0, column=1, sticky="w")

        tk.Label(
            detail_frame,
            text="Near-Fall Rules",
            bg=COLORS["bg"],
            fg=COLORS["text_secondary"],
            font=FONTS["small"],
            anchor="e",
        ).grid(row=1, column=0, sticky="e", padx=(0, 8))

        self._nf_label = tk.Label(
            detail_frame,
            text="—",
            bg=COLORS["bg"],
            fg=COLORS["text_primary"],
            font=(FONTS["body"][0], FONTS["body"][1], "bold"),
            anchor="w",
            width=14,
        )
        self._nf_label.grid(row=1, column=1, sticky="w")

        # ── Post-alert overlay (hidden until a fall is assessed) ───────────
        self._overlay = tk.Frame(self, bg=COLORS["danger"], bd=0)
        # Overlay is placed over the entire screen via place() when needed

    # -----------------------------------------------------------------------
    # Monitoring toggle
    # -----------------------------------------------------------------------

    def _toggle_monitoring(self) -> None:
        if self._running:
            self._stop_monitoring()
        else:
            self._start_monitoring()

    def _start_monitoring(self) -> None:
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            self._set_badge("stopped")
            self._rf_label.configure(text="Camera error")
            return

        self._pipeline     = DetectionPipeline(draw_skeleton=True, show_debug_rules=False)
        self._event_logger = EventLogger()
        self._running      = True

        self._start_btn.configure(
            text="■  Stop Monitoring",
            bg=COLORS["danger"],
        )
        self._set_badge("monitoring")
        self._rf_label.configure(text="—")
        self._nf_label.configure(text="—")

        # Start capture thread
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        # Start polling the frame queue
        self._poll()

    def _stop_monitoring(self) -> None:
        self._running = False

        if self._cap:
            self._cap.release()
            self._cap = None

        if self._pipeline:
            self._pipeline.close()
            self._pipeline = None

        if self._event_logger:
            self._event_logger.reset()
            self._event_logger = None

        self._start_btn.configure(
            text="▶  Start Monitoring",
            bg=COLORS["success"],
        )
        self._set_badge("stopped")
        self._rf_label.configure(text="—")
        self._nf_label.configure(text="—")
        self._draw_idle_canvas()

    # -----------------------------------------------------------------------
    # Capture loop (background thread)
    # -----------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """
        Reads frames from the camera and runs DetectionPipeline on each.
        Puts FrameResult objects into the queue for the main thread to consume.
        Pauses automatically while _in_assessment is True.
        """
        while self._running:
            if self._in_assessment:
                time.sleep(0.05)
                continue

            ret, frame = self._cap.read()
            if not ret:
                break

            # Process the real frame for pose estimation and classification,
            # then draw skeleton onto a blank background (no raw video).
            result = self._pipeline.process_frame(frame)

            h, w = frame.shape[:2]
            blank = np.full((h, w, 3), 240, dtype=np.uint8)  # light grey background
            self._pipeline._draw_pose(frame, blank, result.pose_landmarks)
            self._pipeline._draw_labels(
                blank,
                result.rf_status,
                result.near_fall_status,
                result.debug_rules,
            )

            # Feed skeleton frame (not raw) to event logger so saved clips
            # only contain the skeleton, preserving privacy.
            if self._event_logger:
                clip_path = self._event_logger.add_frame(blank)
                if clip_path:
                    # Do NOT call self.after() here — this is a background thread.
                    # Put the path in a queue; _poll() drains it on the main thread.
                    self._clip_queue.put_nowait(str(clip_path))
            result = result.__class__(
                rf_status        = result.rf_status,
                near_fall_status = result.near_fall_status,
                alert            = result.alert,
                pose_landmarks   = result.pose_landmarks,
                debug_rules      = result.debug_rules,
                annotated_frame  = blank,
            )

            # Drop frame if queue is full (don't block the capture thread)
            try:
                self._frame_queue.put_nowait(result)
            except queue.Full:
                pass

    # -----------------------------------------------------------------------
    # Main thread polling
    # -----------------------------------------------------------------------

    def _poll(self) -> None:
        """
        Called every POLL_MS ms on the main thread.
        Drains the frame queue and updates the UI.
        Schedules itself again if monitoring is still running.
        """
        if not self._running:
            return

        try:
            result: FrameResult = self._frame_queue.get_nowait()
            self._handle_result(result)
        except queue.Empty:
            pass

        # Drain any completed clip paths — safe here because we are on the main thread
        try:
            while True:
                clip_path = self._clip_queue.get_nowait()
                self._on_clip_ready(clip_path)
        except queue.Empty:
            pass

        self.after(POLL_MS, self._poll)

    # -----------------------------------------------------------------------
    # Result handling
    # -----------------------------------------------------------------------

    def _handle_result(self, result: FrameResult) -> None:
        """Process a single FrameResult on the main thread."""

        # ── Update skeleton canvas ─────────────────────────────────────────
        self._draw_skeleton(result.annotated_frame)

        # ── Update detail labels ───────────────────────────────────────────
        self._rf_label.configure(text=result.rf_status.upper())
        self._nf_label.configure(
            text=result.near_fall_status.upper()
            if result.near_fall_status != "no_event"
            else "—"
        )

        # ── Update status badge ────────────────────────────────────────────
        if result.rf_status == "fall":
            self._set_badge("fall")
        elif result.rf_status == "confirming":
            self._set_badge("confirming")
        elif result.near_fall_status == "near_fall":
            self._set_badge("near_fall")
        elif result.pose_landmarks is None:
            self._set_badge("no_pose")
        else:
            self._set_badge("monitoring")

        # ── Log near-falls ─────────────────────────────────────────────────
        if result.near_fall_status == "near_fall":
            self._app.log_event(
                "near_fall",
                f"Near-fall detected. Rules fired: {', '.join(result.debug_rules) or 'n/a'}",
            )

        # ── Trigger assessment on confirmed fall ───────────────────────────
        if result.rf_status == "fall":
            now = time.time()
            print(f"[DEBUG] fall detected | in_assessment={self._in_assessment} | cooldown_remaining={max(0, self._fall_cooldown - (now - self._last_fall_time)):.1f}s")
            if not self._in_assessment and (now - self._last_fall_time) > self._fall_cooldown:
                self._last_fall_time = now
                self._in_assessment  = True
                # Start post-fall clip capture
                if self._event_logger:
                    self._event_logger.on_fall_detected()
                # Log the fall — store ref so we can patch clip_path when clip is ready
                entry = self._app.log_event(
                    "fall",
                    "Fall confirmed by RF classifier. Starting assessment.",
                )
                self._pending_clip_entry = entry
                print("[DEBUG] scheduling _start_assessment")
                self.after(0, self._start_assessment)

    # -----------------------------------------------------------------------
    # Assessment (runs on main thread — pyttsx3 + sounddevice requirement)
    # -----------------------------------------------------------------------

    def _start_assessment(self) -> None:
        """
        Pause monitoring, run the voice assessment pipeline, then show
        the post-alert overlay.  Must run on the main thread.
        """
        print("[DEBUG] _start_assessment called")
        self._in_assessment = True
        self._set_badge("assessment")

        # Get config from setup screen — force instantiation if not yet visited
        self._app.show_screen("setup") if self._app.get_screen("setup") is None else None
        self._app.show_screen("monitoring")   # bring monitoring back to front
        setup  = self._app.get_screen("setup")
        config = setup.get_config() if setup else None
        print(f"[DEBUG] config={config}")

        if config is None:
            self._app.log_event(
                "info",
                "Fall detected but setup is incomplete — "
                "please configure emergency contacts in the Setup tab.",
            )
            self._in_assessment = False
            self._set_badge("monitoring")
            return

        try:
            print("[DEBUG] calling run_assessment")
            result = run_assessment(
                config=config,
                on_status=lambda msg: (print(f"[ASSESSMENT] {msg}"), self._app.log_event("assessment", msg)),
                test_mode=False,
            )
            self._last_result = result
            self._app.log_event(
                "assessment",
                f"Assessment complete — outcome: {result.outcome.value} | "
                f"alert sent: {result.alert_sent}",
            )
            self._show_post_alert_overlay(result)

        except Exception as exc:
            import traceback
            print(f"[DEBUG] Assessment exception: {exc}")
            traceback.print_exc()
            self._app.log_event("info", f"Assessment error: {exc}")
            self._in_assessment = False
            self._set_badge("monitoring")

    # -----------------------------------------------------------------------
    # Clip ready callback
    # -----------------------------------------------------------------------

    def _on_clip_ready(self, clip_path: str) -> None:
        """
        Called on the main thread when EventLogger finishes saving a clip.
        Patches the pending fall log entry with the clip path so the
        event log screen can show a play button.
        """
        if self._pending_clip_entry is not None:
            self._pending_clip_entry["clip_path"] = clip_path
            self._pending_clip_entry["detail"] += f"  Clip saved: {clip_path}"
            # Notify the log screen to refresh that row
            log_screen = self._app.get_screen("log")
            if log_screen is not None:
                log_screen.update_entry(self._pending_clip_entry)
            self._pending_clip_entry = None

    # -----------------------------------------------------------------------
    # Post-alert overlay
    # -----------------------------------------------------------------------

    def _show_post_alert_overlay(self, result: AssessmentResult) -> None:
        """
        Cover the screen with a post-alert summary.
        Monitoring continues in the background but the overlay is on top.
        The user must manually dismiss it.
        """
        # Clear any previous overlay content
        for w in self._overlay.winfo_children():
            w.destroy()

        outcome  = result.outcome.value.replace("_", " ").upper()
        sent     = result.alert_sent
        successes = sum(1 for r in result.alert_results if r.success) if sent else 0
        total     = len(result.alert_results) if sent else 0

        # ── Overlay content ────────────────────────────────────────────────
        tk.Label(
            self._overlay,
            text="⚠  FALL DETECTED",
            bg=COLORS["danger"],
            fg="#FFFFFF",
            font=(FONTS["heading"][0], FONTS["heading"][1], "bold"),
            pady=24,
        ).pack(fill=tk.X)

        tk.Frame(self._overlay, bg="#FFFFFF", height=2).pack(fill=tk.X)

        body = tk.Frame(self._overlay, bg=COLORS["surface"])
        body.pack(fill=tk.BOTH, expand=True, padx=40, pady=32)

        tk.Label(
            body,
            text=f"Assessment outcome:  {outcome}",
            bg=COLORS["surface"],
            fg=COLORS["text_primary"],
            font=(FONTS["subheading"][0], FONTS["subheading"][1], "bold"),
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 16))

        if sent:
            alert_color = COLORS["success"] if successes == total else COLORS["warning"]
            tk.Label(
                body,
                text=f"Emergency contacts notified:  {successes} of {total} succeeded",
                bg=COLORS["surface"],
                fg=alert_color,
                font=FONTS["body"],
                anchor="w",
            ).pack(fill=tk.X, pady=(0, 8))

            for r in result.alert_results:
                icon   = "✓" if r.success else "✗"
                colour = COLORS["success"] if r.success else COLORS["danger"]
                tk.Label(
                    body,
                    text=f"    {icon}  {r.action}",
                    bg=COLORS["surface"],
                    fg=colour,
                    font=FONTS["small"],
                    anchor="w",
                ).pack(fill=tk.X)
        else:
            tk.Label(
                body,
                text="No emergency alert was sent.",
                bg=COLORS["surface"],
                fg=COLORS["text_secondary"],
                font=FONTS["body"],
                anchor="w",
            ).pack(fill=tk.X)

        tk.Label(
            body,
            text=f"Time: {result.timestamp[:19].replace('T', '  ')}",
            bg=COLORS["surface"],
            fg=COLORS["text_secondary"],
            font=FONTS["small"],
            anchor="w",
        ).pack(fill=tk.X, pady=(24, 0))

        # Dismiss button
        tk.Frame(body, bg=COLORS["border"], height=2).pack(fill=tk.X, pady=(32, 0))

        dismiss_btn = tk.Label(
            body,
            text="Dismiss — return to monitoring",
            bg=COLORS["accent"],
            fg="#FFFFFF",
            font=FONTS["button"],
            padx=32,
            pady=16,
            cursor="hand2",
        )
        dismiss_btn.pack(pady=(16, 0))
        dismiss_btn.bind("<Button-1>", lambda e: self._dismiss_overlay())
        dismiss_btn.bind("<Enter>",    lambda e: dismiss_btn.configure(bg=COLORS["accent_hover"]))
        dismiss_btn.bind("<Leave>",    lambda e: dismiss_btn.configure(bg=COLORS["accent"]))

        # Place overlay on top of entire screen
        self._overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._overlay.lift()
        self._overlay_visible = True

    def _dismiss_overlay(self) -> None:
        """Remove the post-alert overlay and resume normal monitoring."""
        self._overlay.place_forget()
        self._overlay_visible = False
        self._in_assessment   = False
        self._set_badge("monitoring")

    # -----------------------------------------------------------------------
    # Canvas drawing
    # -----------------------------------------------------------------------

    def _draw_skeleton(self, frame_bgr: np.ndarray) -> None:
        """
        Convert the annotated BGR frame to a PhotoImage and display it on
        the canvas, scaled to fit CANVAS_W x CANVAS_H.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img       = Image.fromarray(frame_rgb).resize(
            (CANVAS_W, CANVAS_H), Image.BILINEAR
        )
        photo = ImageTk.PhotoImage(img)

        # Keep a reference — without this Tkinter garbage-collects the image
        self._photo_ref = photo
        self._canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    def _draw_idle_canvas(self) -> None:
        """Draw a placeholder message when monitoring is not running."""
        self._canvas.delete("all")
        self._canvas.create_text(
            CANVAS_W // 2,
            CANVAS_H // 2,
            text="Press  ▶  Start Monitoring  to begin",
            fill="#555555",
            font=(FONTS["body"][0], FONTS["body"][1]),
            justify=tk.CENTER,
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _set_badge(self, key: str) -> None:
        style = _STATUS_STYLE.get(key, _STATUS_STYLE["monitoring"])
        self._badge.configure(
            text=style["label"],
            bg=style["bg"],
            fg=style["fg"],
        )

    # -----------------------------------------------------------------------
    # Lifecycle — called by app when navigating away / closing
    # -----------------------------------------------------------------------

    def on_hide(self) -> None:
        """Called when the user navigates away from this screen."""
        # Don't stop monitoring just because they switched tabs
        pass

    def on_close(self) -> None:
        """Called when the app window is closing."""
        self._stop_monitoring()