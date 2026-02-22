"""
ui/app.py

Tkinter root window and screen manager for the fall detection system.

Manages switching between screens:
    - SetupScreen    — emergency contacts, credentials, test button
    - MonitoringScreen — live feed + skeleton overlay + status badge
    - EventLogScreen — scrollable log viewer

Usage
-----
    from ui.app import App
    app = App()
    app.mainloop()
"""

from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from typing import Type

# ---------------------------------------------------------------------------
# Design constants — shared across all screens
# ---------------------------------------------------------------------------

COLORS = {
    # High-contrast light theme — WCAG AA compliant throughout
    "bg":             "#F5F5F0",   # warm off-white — easier on eyes than pure white
    "surface":        "#FFFFFF",   # card background
    "surface_raised": "#EAEAE4",   # input field background
    "border":         "#AAAAAA",   # visible border — 4.5:1 contrast on white
    "accent":         "#0052CC",   # deep blue — 7:1 on white, WCAG AAA
    "accent_hover":   "#003D99",
    "danger":         "#CC0000",   # deep red — 5.9:1 on white
    "danger_hover":   "#990000",
    "success":        "#006622",   # deep green — 7:1 on white
    "warning":        "#8A5000",   # deep amber — readable on light bg
    "text_primary":   "#111111",   # near-black — 16:1 contrast on white
    "text_secondary": "#444444",   # dark grey — 9.7:1 on white
    "text_disabled":  "#767676",   # minimum WCAG AA on white
    "input_focus":    "#0052CC",   # focus ring color
}

FONTS = {
    # Scaled from a 16px base — proportions hold across screen sizes
    "heading":    ("Georgia", 28, "bold"),     # page title
    "subheading": ("Georgia", 20, "bold"),     # section title
    "label":      ("Helvetica Neue", 16),      # field labels
    "label_bold": ("Helvetica Neue", 16, "bold"),
    "body":       ("Helvetica Neue", 16),      # body / input text
    "small":      ("Helvetica Neue", 14),      # captions / hints
    "button":     ("Helvetica Neue", 16, "bold"),
    "nav":        ("Helvetica Neue", 30, "bold"),
    "mono":       ("Courier", 14),
}

PADDING = {
    "screen":  24,   # outer screen margin
    "section": 16,   # between sections
    "item":     8,   # between items within a section
}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class App(tk.Tk):
    """
    Root Tkinter window. Owns the screen container and handles screen
    transitions. All screens are instantiated lazily on first navigation.

    Screens register themselves by subclassing tk.Frame and being passed
    to show_screen(). The currently visible screen fills the container.
    """

    def __init__(self):
        super().__init__()

        self.title("Fall Detection System")
        self.geometry("960x700")
        self.minsize(800, 600)
        self.configure(bg=COLORS["bg"])

        # Center window on screen
        self.update_idletasks()
        x = (self.winfo_screenwidth()  - 960) // 2
        y = (self.winfo_screenheight() - 700) // 2
        self.geometry(f"960x700+{x}+{y}")

        # --- Navigation bar ---
        self._nav = _NavBar(self, on_navigate=self._on_nav)
        self._nav.pack(side=tk.TOP, fill=tk.X)
        # Bottom border on nav
        tk.Frame(self, bg=COLORS["border"], height=2).pack(side=tk.TOP, fill=tk.X)

        # --- Screen container — screens are stacked here ---
        self._container = tk.Frame(self, bg=COLORS["bg"])
        self._container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._container.grid_rowconfigure(0, weight=1)
        self._container.grid_columnconfigure(0, weight=1)

        self._screens: dict[str, tk.Frame] = {}
        self._current_screen: str | None = None

        # Shared event log — written by MonitoringScreen, read by EventLogScreen
        # Each entry: {"time": str, "type": "fall"|"near_fall"|"assessment", "detail": str}
        self.event_log: list[dict] = self._load_event_log()

        # Start on the setup screen
        self.show_screen("setup")

        # Clean up monitoring on window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self) -> None:
        monitor = self._screens.get("monitoring")
        if monitor is not None:
            monitor.on_close()
        self._save_event_log()
        self.destroy()

    # -----------------------------------------------------------------------
    # Event log persistence
    # -----------------------------------------------------------------------

    # Always save next to config.json in the project root,
    # regardless of the working directory when the app is launched.
    LOG_FILE = str(__import__("pathlib").Path(__file__).parent.parent / "event_log.json")

    def _load_event_log(self) -> list[dict]:
        """Read persisted event log from disk. Returns empty list if not found."""
        import json, os
        if not os.path.exists(self.LOG_FILE):
            return []
        try:
            with open(self.LOG_FILE) as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_event_log(self) -> None:
        """Write the current event log to disk."""
        import json
        try:
            with open(self.LOG_FILE, "w") as f:
                json.dump(self.event_log, f, indent=2)
        except Exception:
            pass  # Don't crash on close if save fails

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def show_screen(self, name: str) -> None:
        """
        Navigate to a screen by name.

        Screens are instantiated lazily on first visit. Subsequent visits
        re-use the existing instance (state is preserved).

        Parameters
        ----------
        name : str
            One of: "setup", "monitoring", "log"
        """
        if name not in self._screens:
            self._screens[name] = self._build_screen(name)

        screen = self._screens[name]
        screen.tkraise()
        self._current_screen = name
        self._nav.set_active(name)

    def get_screen(self, name: str) -> tk.Frame | None:
        """Return an already-instantiated screen by name, or None."""
        return self._screens.get(name)

    def log_event(self, event_type: str, detail: str = "") -> None:
        """
        Append an event to the shared log and notify EventLogScreen if visible.

        Parameters
        ----------
        event_type : str
            One of: "fall", "near_fall", "assessment", "info"
        detail : str
            Human-readable description of the event.
        """
        from datetime import datetime
        entry = {
            "time":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type":   event_type,
            "detail": detail,
        }
        self.event_log.append(entry)

        # Live-push to event log screen if it's already instantiated
        log_screen = self._screens.get("log")
        if log_screen is not None:
            log_screen.push_event(entry)

        # Return the entry so callers can mutate it later (e.g. add clip_path)
        return entry

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _build_screen(self, name: str) -> tk.Frame:
        """Instantiate and grid a screen frame inside the container."""
        # Import here to avoid circular imports at module load time
        if name == "setup":
            from ui.setup_screen import SetupScreen
            screen = SetupScreen(self._container, app=self)
        elif name == "monitoring":
            from ui.monitoring_screen import MonitoringScreen
            screen = MonitoringScreen(self._container, app=self)
        elif name == "log":
            from ui.event_log_screen import EventLogScreen
            screen = EventLogScreen(self._container, app=self)
        else:
            raise ValueError(f"Unknown screen name: '{name}'")

        screen.grid(row=0, column=0, sticky="nsew")
        return screen

    def _on_nav(self, name: str) -> None:
        self.show_screen(name)


# ---------------------------------------------------------------------------
# Navigation bar
# ---------------------------------------------------------------------------

class _NavBar(tk.Frame):
    """
    Top navigation bar with three screen buttons and the app title.
    """

    _SCREENS = [
        ("setup",      "⚙  Setup"),
        ("monitoring", "◉  Monitor"),
        ("log",        "≡  Event Log"),
    ]

    def __init__(self, parent: tk.Widget, on_navigate):
        super().__init__(parent, bg=COLORS["surface"], height=72, relief=tk.FLAT)
        self.pack_propagate(False)
        self._on_navigate = on_navigate
        self._buttons: dict[str, tk.Label] = {}

        # App title
        tk.Label(
            self,
            text="Fall Detection System",
            bg=COLORS["surface"],
            fg=COLORS["text_primary"],
            font=FONTS["nav"],
            padx=24,
        ).pack(side=tk.LEFT)

        # Separator
        tk.Frame(self, bg=COLORS["border"], width=1).pack(
            side=tk.LEFT, fill=tk.Y, pady=8
        )

        # Nav buttons
        for name, label in self._SCREENS:
            btn = tk.Label(
                self,
                text=label,
                bg=COLORS["surface"],
                fg=COLORS["text_secondary"],
                font=FONTS["nav"],
                padx=20,
                pady=4,
                cursor="hand2",
            )
            btn.pack(side=tk.LEFT)
            btn.bind("<Button-1>", lambda e, n=name: self._on_navigate(n))
            btn.bind("<Enter>",    lambda e, b=btn: b.configure(fg=COLORS["text_primary"]))
            btn.bind("<Leave>",    lambda e, b=btn, n=name: self._restore(b, n))
            self._buttons[name] = btn

    def set_active(self, name: str) -> None:
        """Highlight the active screen's nav button."""
        for n, btn in self._buttons.items():
            if n == name:
                btn.configure(fg=COLORS["accent"], font=FONTS["nav"])
            else:
                btn.configure(fg=COLORS["text_secondary"], font=FONTS["nav"])

    def _restore(self, btn: tk.Label, name: str) -> None:
        if self._buttons.get(name) == btn:
            is_active = any(
                n == name and btn.cget("font") == str(FONTS["label_bold"])
                for n, _ in self._SCREENS
            )
            btn.configure(fg=COLORS["accent"] if btn.cget("fg") == COLORS["accent"] else COLORS["text_secondary"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = App()
    app.mainloop()