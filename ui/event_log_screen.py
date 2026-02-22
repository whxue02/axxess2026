"""
ui/event_log_screen.py

Scrollable event log screen for the fall detection system.

Shows a chronological list of all falls, near-falls, assessment outcomes,
and system info messages. New events are pushed live from the monitoring
screen via app.log_event() -> push_event().

Entry types and their visual treatment:
    fall        — red left border, bold label
    near_fall   — amber left border, bold label
    assessment  — blue left border
    info        — grey left border, muted text
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from typing import TYPE_CHECKING

import tkinter as tk

if TYPE_CHECKING:
    from ui.app import App

from ui.app import COLORS, FONTS, PADDING

# ---------------------------------------------------------------------------
# Per-type visual config
# ---------------------------------------------------------------------------

_TYPE_STYLE = {
    "fall": {
        "border":  COLORS["danger"],
        "label":   "FALL",
        "label_fg": COLORS["danger"],
        "bg":       "#FFF5F5",
    },
    "near_fall": {
        "border":  COLORS["warning"],
        "label":   "NEAR FALL",
        "label_fg": COLORS["warning"],
        "bg":       "#FFFBF0",
    },
    "assessment": {
        "border":  COLORS["accent"],
        "label":   "ASSESSMENT",
        "label_fg": COLORS["accent"],
        "bg":       COLORS["surface"],
    },
    "info": {
        "border":  COLORS["border"],
        "label":   "INFO",
        "label_fg": COLORS["text_disabled"],
        "bg":       COLORS["surface"],
    },
}

_DEFAULT_STYLE = _TYPE_STYLE["info"]


# ---------------------------------------------------------------------------
# Event log screen
# ---------------------------------------------------------------------------

class EventLogScreen(tk.Frame):
    """
    Scrollable log of all fall-related events.

    On first display, back-fills from app.event_log so events that happened
    before this screen was first visited are still shown.

    New events are pushed live by app.log_event() calling push_event().
    """

    def __init__(self, parent: tk.Widget, app: "App"):
        super().__init__(parent, bg=COLORS["bg"])
        self._app         = app
        self._entry_count = 0
        # Maps entry id() -> (wrapper Frame, entry dict) for live updates
        self._entry_widgets: dict[int, tk.Frame] = {}
        self._build()
        # Back-fill any events that were logged before this screen was opened
        self.after(0, self._backfill)

    # -----------------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------------

    def _build(self) -> None:
        # ── Header row ─────────────────────────────────────────────────────
        header = tk.Frame(self, bg=COLORS["bg"])
        header.pack(fill=tk.X, padx=40, pady=(28, 0))

        tk.Label(
            header,
            text="Event Log",
            bg=COLORS["bg"],
            fg=COLORS["text_primary"],
            font=(FONTS["heading"][0], FONTS["heading"][1], "bold"),
            anchor="w",
        ).pack(side=tk.LEFT)

        self._count_label = tk.Label(
            header,
            text="No events yet",
            bg=COLORS["bg"],
            fg=COLORS["text_secondary"],
            font=FONTS["small"],
            anchor="e",
        )
        self._count_label.pack(side=tk.RIGHT)

        tk.Label(
            self,
            text="Falls, near-falls, and assessment outcomes are recorded here in real time.",
            bg=COLORS["bg"],
            fg=COLORS["text_secondary"],
            font=FONTS["small"],
            anchor="w",
        ).pack(fill=tk.X, padx=40, pady=(4, 16))

        # Clear button
        clear_btn = tk.Label(
            self,
            text="Clear log",
            bg=COLORS["bg"],
            fg=COLORS["text_disabled"],
            font=(FONTS["small"][0], FONTS["small"][1]),
            anchor="w",
            cursor="hand2",
            padx=40,
        )
        clear_btn.pack(anchor="w")
        clear_btn.bind("<Button-1>", lambda e: self._clear())
        clear_btn.bind("<Enter>",    lambda e: clear_btn.configure(fg=COLORS["danger"]))
        clear_btn.bind("<Leave>",    lambda e: clear_btn.configure(fg=COLORS["text_disabled"]))

        tk.Frame(self, bg=COLORS["border"], height=2).pack(fill=tk.X, padx=40, pady=(12, 0))

        # ── Scrollable event list ───────────────────────────────────────────
        self._canvas = tk.Canvas(self, bg=COLORS["bg"], highlightthickness=0)
        scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._list_frame = tk.Frame(self._canvas, bg=COLORS["bg"])
        self._list_frame.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        self._canvas_window = self._canvas.create_window(
            (0, 0), window=self._list_frame, anchor="nw"
        )
        self._canvas.bind(
            "<Configure>",
            lambda e: self._canvas.itemconfig(self._canvas_window, width=e.width),
        )

        # Mousewheel scrolling
        self._canvas.bind_all("<MouseWheel>", lambda e: self._canvas.yview_scroll(-1 * (e.delta // 120), "units"))
        self._canvas.bind_all("<Button-4>",   lambda e: self._canvas.yview_scroll(-1, "units"))
        self._canvas.bind_all("<Button-5>",   lambda e: self._canvas.yview_scroll(1,  "units"))

        # Empty state label — shown when there are no events
        self._empty_label = tk.Label(
            self._list_frame,
            text="No events recorded yet.\nEvents will appear here when monitoring is active.",
            bg=COLORS["bg"],
            fg=COLORS["text_disabled"],
            font=FONTS["body"],
            justify=tk.CENTER,
            pady=60,
        )
        self._empty_label.pack(expand=True)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def push_event(self, entry: dict) -> None:
        """
        Append a single event entry to the log.
        Safe to call from any thread via app.log_event() → self.after().

        entry keys: time (str), type (str), detail (str)
        """
        self.after(0, lambda: self._add_row(entry))

    # -----------------------------------------------------------------------
    # Row rendering
    # -----------------------------------------------------------------------

    def _add_row(self, entry: dict) -> None:
        """Render one event row and scroll to it."""
        # Hide empty state on first event
        if self._entry_count == 0:
            self._empty_label.pack_forget()

        self._entry_count += 1
        self._update_count_label()

        style     = _TYPE_STYLE.get(entry["type"], _DEFAULT_STYLE)
        row_bg    = style["bg"]
        border_c  = style["border"]

        # Find the first PACKED child before creating wrapper.
        # pack_forget() removes widgets from the pack manager, so winfo_children()
        # can include hidden widgets that before= cannot reference.
        first_packed = next(
            (w for w in self._list_frame.winfo_children()
             if w.winfo_manager() == "pack"),
            None,
        )
        wrapper = tk.Frame(self._list_frame, bg=COLORS["bg"])
        if first_packed:
            wrapper.pack(fill=tk.X, padx=40, pady=(0, 8), before=first_packed)
        else:
            wrapper.pack(fill=tk.X, padx=40, pady=(0, 8))

        border = tk.Frame(wrapper, bg=border_c, width=5)
        border.pack(side=tk.LEFT, fill=tk.Y)

        card = tk.Frame(
            wrapper,
            bg=row_bg,
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        card.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # ── Card interior ─────────────────────────────────────────────────
        interior = tk.Frame(card, bg=row_bg)
        interior.pack(fill=tk.X, padx=16, pady=12)

        # Top row: type badge + timestamp
        top = tk.Frame(interior, bg=row_bg)
        top.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            top,
            text=style["label"],
            bg=row_bg,
            fg=style["label_fg"],
            font=(FONTS["small"][0], FONTS["small"][1], "bold"),
            anchor="w",
        ).pack(side=tk.LEFT)

        tk.Label(
            top,
            text=entry.get("time", ""),
            bg=row_bg,
            fg=COLORS["text_disabled"],
            font=FONTS["mono"],
            anchor="e",
        ).pack(side=tk.RIGHT)

        # Detail text
        detail = entry.get("detail", "").strip()
        if detail:
            tk.Label(
                interior,
                text=detail,
                bg=row_bg,
                fg=COLORS["text_primary"] if entry["type"] in ("fall", "near_fall") else COLORS["text_secondary"],
                font=FONTS["body"],
                anchor="w",
                wraplength=700,
                justify=tk.LEFT,
            ).pack(fill=tk.X)

        # Store widget ref keyed by entry object id for later updates
        self._entry_widgets[id(entry)] = (wrapper, entry)

        # Play button — shown immediately if clip already exists, or added later
        if entry.get("clip_path"):
            self._add_play_button(interior, row_bg, entry["clip_path"])

        # Scroll to top so the newest event (just inserted at top) is visible
        self._canvas.update_idletasks()
        self._canvas.yview_moveto(0.0)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def update_entry(self, entry: dict) -> None:
        """
        Called when a clip_path is added to an existing log entry.
        Finds the rendered row and appends a play button to it.
        """
        self.after(0, lambda: self._patch_row(entry))

    def _patch_row(self, entry: dict) -> None:
        """Add a play button to an already-rendered row."""
        result = self._entry_widgets.get(id(entry))
        if result is None:
            return
        wrapper, _ = result
        clip_path = entry.get("clip_path")
        if not clip_path:
            return
        # Find the interior frame (card > interior)
        try:
            card    = [c for c in wrapper.winfo_children() if isinstance(c, tk.Frame)][-1]
            interior = [c for c in card.winfo_children() if isinstance(c, tk.Frame)][0]
            row_bg  = card.cget("bg")
            self._add_play_button(interior, row_bg, clip_path)
        except (IndexError, tk.TclError):
            pass

    def _add_play_button(self, parent: tk.Frame, bg: str, clip_path: str) -> None:
        """Append a play button that opens the clip in the system video player."""
        btn_row = tk.Frame(parent, bg=bg)
        btn_row.pack(fill=tk.X, pady=(8, 0))

        tk.Frame(btn_row, bg=COLORS["border"], height=1).pack(fill=tk.X, pady=(0, 8))

        play_btn = tk.Label(
            btn_row,
            text="▶  Play fall clip",
            bg=COLORS["accent"],
            fg="#FFFFFF",
            font=(FONTS["small"][0], FONTS["small"][1], "bold"),
            padx=16,
            pady=8,
            cursor="hand2",
        )
        play_btn.pack(side=tk.LEFT)
        play_btn.bind("<Button-1>", lambda e, p=clip_path: self._open_clip(p))
        play_btn.bind("<Enter>",    lambda e: play_btn.configure(bg=COLORS["accent_hover"]))
        play_btn.bind("<Leave>",    lambda e: play_btn.configure(bg=COLORS["accent"]))

        tk.Label(
            btn_row,
            text=os.path.basename(clip_path),
            bg=bg,
            fg=COLORS["text_disabled"],
            font=FONTS["mono"],
            padx=12,
        ).pack(side=tk.LEFT)

    def _open_clip(self, clip_path: str) -> None:
        """Open the clip in the OS default video player."""
        if not os.path.exists(clip_path):
            return
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", clip_path])
            elif sys.platform == "win32":
                os.startfile(clip_path)
            else:
                subprocess.Popen(["xdg-open", clip_path])
        except Exception:
            pass

    def _backfill(self) -> None:
        """Render all events that were logged before this screen was opened.
        Iterates in reverse so the oldest entry ends up at the bottom.
        """
        for entry in reversed(self._app.event_log):
            self._add_row(entry)

    def _clear(self) -> None:
        """Remove all rows from the display, reset counter, and clear persisted log."""
        for widget in self._list_frame.winfo_children():
            widget.destroy()
        self._entry_count = 0
        self._update_count_label()
        # Also clear the in-memory log so it doesn't get re-saved on close
        self._app.event_log.clear()
        # Restore empty state label
        self._empty_label = tk.Label(
            self._list_frame,
            text="No events recorded yet.\nEvents will appear here when monitoring is active.",
            bg=COLORS["bg"],
            fg=COLORS["text_disabled"],
            font=FONTS["body"],
            justify=tk.CENTER,
            pady=60,
        )
        self._empty_label.pack(expand=True)

    def _update_count_label(self) -> None:
        if self._entry_count == 0:
            self._count_label.configure(text="No events yet")
        elif self._entry_count == 1:
            self._count_label.configure(text="1 event")
        else:
            self._count_label.configure(text=f"{self._entry_count} events")