"""
ui/setup_screen.py

Setup screen for the fall detection system.

Sections:
    1. User profile       — monitored user's name
    2. Emergency contacts — add / remove contacts (name + phone)
    3. Actions            — Send Test Alert, Save Configuration

Design principles:
    - WCAG AA contrast throughout (all text >= 4.5:1, large text >= 3:1)
    - Minimum 16pt body text; labels and inputs scale with screen size
    - Touch-friendly tap targets (min 44px height)
    - Clear focus indicators on all interactive elements
    - Twilio credentials come from .env — not surfaced here
    - Family member configures this once; elderly user never touches it
"""

from __future__ import annotations

import tkinter as tk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ui.app import App

from ui.app import COLORS, FONTS, PADDING
from response import run_assessment, AlertConfig, EmergencyContact


# ---------------------------------------------------------------------------
# Placeholder helper
# ---------------------------------------------------------------------------

def _add_placeholder(entry: tk.Entry, var: tk.StringVar, placeholder: str) -> None:
    """Show hint text when field is empty and unfocused."""
    def _on_focus_in(e):
        if var.get() == placeholder:
            var.set("")
            entry.configure(fg=COLORS["text_primary"])

    def _on_focus_out(e):
        if not var.get().strip():
            var.set(placeholder)
            entry.configure(fg=COLORS["text_disabled"])

    var.set(placeholder)
    entry.configure(fg=COLORS["text_disabled"])
    entry.bind("<FocusIn>",  _on_focus_in)
    entry.bind("<FocusOut>", _on_focus_out)


# ---------------------------------------------------------------------------
# Accessible button
# ---------------------------------------------------------------------------

def _make_button(
    parent: tk.Widget,
    text: str,
    command,
    bg: str,
    fg: str,
    hover_bg: str,
    font=None,
    padx: int = 32,
    pady: int = 14,
) -> tk.Label:
    """
    Large, high-contrast button with visible focus ring and hover state.
    Meets WCAG 2.1 minimum 44px touch target.
    """
    font = font or FONTS["button"]
    btn = tk.Label(
        parent,
        text=text,
        bg=bg,
        fg=fg,
        font=font,
        padx=padx,
        pady=pady,
        cursor="hand2",
        relief=tk.FLAT,
    )
    btn.bind("<Button-1>", lambda e: command())
    btn.bind("<Enter>",    lambda e: btn.configure(bg=hover_bg))
    btn.bind("<Leave>",    lambda e: btn.configure(bg=bg))
    btn.bind("<Return>",   lambda e: command())
    btn.bind("<space>",    lambda e: command())
    return btn


# ---------------------------------------------------------------------------
# Contact row
# ---------------------------------------------------------------------------

class _ContactRow(tk.Frame):
    """One emergency contact with name, phone, and remove button."""

    PLACEHOLDER_NAME  = "Contact's full name"
    PLACEHOLDER_PHONE = "+12125551234"

    def __init__(self, parent: tk.Widget, on_remove, index: int, scale: float = 1.0):
        super().__init__(parent, bg=COLORS["surface"])
        self.index  = index
        self._scale = scale
        self.name_var  = tk.StringVar()
        self.phone_var = tk.StringVar()
        self._build(on_remove)

    def _build(self, on_remove) -> None:
        s = self._scale

        # Row number
        tk.Label(
            self,
            text=f"{self.index + 1}.",
            bg=COLORS["surface"],
            fg=COLORS["text_secondary"],
            font=(FONTS["label"][0], int(FONTS["label"][1] * s), "bold"),
            width=3,
            anchor="e",
        ).pack(side=tk.LEFT, padx=(0, 16))

        # Name column
        name_col = tk.Frame(self, bg=COLORS["surface"])
        name_col.pack(side=tk.LEFT, padx=(0, 20))

        tk.Label(
            name_col,
            text="Name",
            bg=COLORS["surface"],
            fg=COLORS["text_secondary"],
            font=(FONTS["small"][0], int(FONTS["small"][1] * s)),
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 4))

        name_entry = tk.Entry(
            name_col,
            textvariable=self.name_var,
            font=(FONTS["body"][0], int(FONTS["body"][1] * s)),
            bg=COLORS["surface_raised"],
            fg=COLORS["text_primary"],
            insertbackground=COLORS["text_primary"],
            relief=tk.FLAT,
            highlightbackground=COLORS["border"],
            highlightcolor=COLORS["input_focus"],
            highlightthickness=2,
            width=22,
        )
        name_entry.pack(ipady=int(10 * s))
        _add_placeholder(name_entry, self.name_var, self.PLACEHOLDER_NAME)

        # Phone column
        phone_col = tk.Frame(self, bg=COLORS["surface"])
        phone_col.pack(side=tk.LEFT, padx=(0, 20))

        tk.Label(
            phone_col,
            text="Phone number (E.164 format)",
            bg=COLORS["surface"],
            fg=COLORS["text_secondary"],
            font=(FONTS["small"][0], int(FONTS["small"][1] * s)),
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 4))

        phone_entry = tk.Entry(
            phone_col,
            textvariable=self.phone_var,
            font=(FONTS["body"][0], int(FONTS["body"][1] * s)),
            bg=COLORS["surface_raised"],
            fg=COLORS["text_primary"],
            insertbackground=COLORS["text_primary"],
            relief=tk.FLAT,
            highlightbackground=COLORS["border"],
            highlightcolor=COLORS["input_focus"],
            highlightthickness=2,
            width=18,
        )
        phone_entry.pack(ipady=int(10 * s))
        _add_placeholder(phone_entry, self.phone_var, self.PLACEHOLDER_PHONE)

        # Remove button — aligned with inputs via spacer label
        rm_col = tk.Frame(self, bg=COLORS["surface"])
        rm_col.pack(side=tk.LEFT)

        tk.Label(
            rm_col, text="",
            bg=COLORS["surface"],
            font=(FONTS["small"][0], int(FONTS["small"][1] * s)),
        ).pack(pady=(0, 4))

        rm = tk.Label(
            rm_col,
            text="Remove",
            bg=COLORS["surface"],
            fg=COLORS["danger"],
            font=(FONTS["small"][0], int(FONTS["small"][1] * s), "bold"),
            cursor="hand2",
            padx=8,
            pady=int(10 * s),
        )
        rm.pack()
        rm.bind("<Button-1>", lambda e: on_remove(self))
        rm.bind("<Enter>",    lambda e: rm.configure(bg=COLORS["danger"], fg="#FFFFFF"))
        rm.bind("<Leave>",    lambda e: rm.configure(bg=COLORS["surface"], fg=COLORS["danger"]))

    def get(self) -> tuple[str, str]:
        name  = self.name_var.get().strip()
        phone = self.phone_var.get().strip()
        if name  == self.PLACEHOLDER_NAME:  name  = ""
        if phone == self.PLACEHOLDER_PHONE: phone = ""
        return name, phone

    def set(self, name: str, phone: str) -> None:
        self.name_var.set(name)
        self.phone_var.set(phone)


# ---------------------------------------------------------------------------
# Setup screen
# ---------------------------------------------------------------------------

class SetupScreen(tk.Frame):
    """
    Configuration screen. Scales fonts and spacing proportionally to the
    window size so the layout is comfortable at any screen size.
    """

    def __init__(self, parent: tk.Widget, app: "App"):
        super().__init__(parent, bg=COLORS["bg"])
        self._app = app
        self._contact_rows: list[_ContactRow] = []
        self._scale = 1.0

        self._user_name_var = tk.StringVar()
        self._status_var    = tk.StringVar(value="")

        self.bind("<Configure>", self._on_resize)
        self._build()

    # -----------------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------------

    def _build(self) -> None:
        self._canvas = tk.Canvas(self, bg=COLORS["bg"], highlightthickness=0)
        scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._inner = tk.Frame(self._canvas, bg=COLORS["bg"])
        self._inner.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        self._canvas_window = self._canvas.create_window(
            (0, 0), window=self._inner, anchor="nw"
        )
        self._canvas.bind("<Configure>", self._on_canvas_resize)

        self._canvas.bind_all("<MouseWheel>", lambda e: self._canvas.yview_scroll(-1 * (e.delta // 120), "units"))
        self._canvas.bind_all("<Button-4>",   lambda e: self._canvas.yview_scroll(-1, "units"))
        self._canvas.bind_all("<Button-5>",   lambda e: self._canvas.yview_scroll(1,  "units"))

        self._populate()

    def _populate(self) -> None:
        """Rebuild all content inside the scrollable frame."""
        for w in self._inner.winfo_children():
            w.destroy()
        self._contact_rows = []

        s = self._scale
        p = int(40 * s)   # outer horizontal padding
        g = int(24 * s)   # gap between sections
        inner = self._inner

        # ── Heading ────────────────────────────────────────────────────────
        tk.Label(
            inner,
            text="System Setup",
            bg=COLORS["bg"],
            fg=COLORS["text_primary"],
            font=(FONTS["heading"][0], int(FONTS["heading"][1] * s), "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=p, pady=(p, 6))


        tk.Frame(inner, bg=COLORS["border"], height=2).pack(fill=tk.X, padx=p, pady=(0, g))

        # ── Section 1: User name ───────────────────────────────────────────
        self._section_heading(inner, "WHO IS BEING MONITORED?", p, s)

        name_card = self._card(inner, p)
        card_body = tk.Frame(name_card, bg=COLORS["surface"])
        card_body.pack(fill=tk.X, padx=int(24 * s), pady=int(20 * s))

        tk.Label(
            card_body,
            text="Full name",
            bg=COLORS["surface"],
            fg=COLORS["text_primary"],
            font=(FONTS["label"][0], int(FONTS["label"][1] * s), "bold"),
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 6))

        name_entry = tk.Entry(
            card_body,
            textvariable=self._user_name_var,
            font=(FONTS["body"][0], int(FONTS["body"][1] * s)),
            bg=COLORS["surface_raised"],
            fg=COLORS["text_primary"],
            insertbackground=COLORS["text_primary"],
            relief=tk.FLAT,
            highlightbackground=COLORS["border"],
            highlightcolor=COLORS["input_focus"],
            highlightthickness=2,
            width=38,
        )
        name_entry.pack(anchor="w", ipady=int(10 * s))
        _add_placeholder(name_entry, self._user_name_var, "e.g. Margaret Smith")

        # ── Section 2: Emergency contacts ─────────────────────────────────
        tk.Frame(inner, bg=COLORS["border"], height=2).pack(fill=tk.X, padx=p, pady=(g, g))

        contacts_header_row = tk.Frame(inner, bg=COLORS["bg"])
        contacts_header_row.pack(fill=tk.X, padx=p, pady=(0, 8))

        tk.Label(
            contacts_header_row,
            text="EMERGENCY CONTACTS",
            bg=COLORS["bg"],
            fg=COLORS["text_secondary"],
            font=(FONTS["small"][0], int(FONTS["small"][1] * s), "bold"),
            anchor="w",
        ).pack(side=tk.LEFT)

        tk.Label(
            contacts_header_row,
            text="All contacts receive a call when an alert fires.",
            bg=COLORS["bg"],
            fg=COLORS["text_secondary"],
            font=(FONTS["small"][0], int(FONTS["small"][1] * s)),
            anchor="e",
        ).pack(side=tk.RIGHT)

        contacts_card = self._card(inner, p)
        self._contacts_container = tk.Frame(contacts_card, bg=COLORS["surface"])
        self._contacts_container.pack(fill=tk.X, padx=int(24 * s), pady=(int(20 * s), 0))

        self._add_contact_row()

        tk.Frame(contacts_card, bg=COLORS["border"], height=1).pack(
            fill=tk.X, padx=int(24 * s), pady=(int(16 * s), 0)
        )

        add_btn = tk.Label(
            contacts_card,
            text="＋  Add another contact",
            bg=COLORS["surface"],
            fg=COLORS["accent"],
            font=(FONTS["label"][0], int(FONTS["label"][1] * s), "bold"),
            anchor="w",
            cursor="hand2",
            padx=int(24 * s),
            pady=int(16 * s),
        )
        add_btn.pack(anchor="w")
        add_btn.bind("<Button-1>", lambda e: self._add_contact_row())
        add_btn.bind("<Enter>",    lambda e: add_btn.configure(fg=COLORS["accent_hover"]))
        add_btn.bind("<Leave>",    lambda e: add_btn.configure(fg=COLORS["accent"]))

        # ── Section 3: Buttons ─────────────────────────────────────────────
        tk.Frame(inner, bg=COLORS["border"], height=2).pack(fill=tk.X, padx=p, pady=(g, g))

        btn_row = tk.Frame(inner, bg=COLORS["bg"])
        btn_row.pack(fill=tk.X, padx=p, pady=(0, int(12 * s)))

        _make_button(
            btn_row,
            text="Send Test Alert",
            command=self._on_test,
            bg=COLORS["accent"],
            fg="#FFFFFF",
            hover_bg=COLORS["accent_hover"],
            font=(FONTS["button"][0], int(FONTS["button"][1] * s), "bold"),
            padx=int(36 * s),
            pady=int(16 * s),
        ).pack(side=tk.LEFT, padx=(0, int(16 * s)))

        _make_button(
            btn_row,
            text="Save Configuration",
            command=self._on_save,
            bg=COLORS["surface_raised"],
            fg=COLORS["text_primary"],
            hover_bg=COLORS["border"],
            font=(FONTS["button"][0], int(FONTS["button"][1] * s), "bold"),
            padx=int(36 * s),
            pady=int(16 * s),
        ).pack(side=tk.LEFT)

        # ── Status message ─────────────────────────────────────────────────
        self._status_label = tk.Label(
            inner,
            textvariable=self._status_var,
            bg=COLORS["bg"],
            fg=COLORS["text_secondary"],
            font=(FONTS["body"][0], int(FONTS["body"][1] * s)),
            anchor="w",
            wraplength=int(860 * s),
            justify=tk.LEFT,
        )
        self._status_label.pack(fill=tk.X, padx=p, pady=(int(8 * s), p))

    # -----------------------------------------------------------------------
    # Widget helpers
    # -----------------------------------------------------------------------

    def _section_heading(self, parent: tk.Widget, text: str, padx: int, s: float) -> None:
        tk.Label(
            parent,
            text=text,
            bg=COLORS["bg"],
            fg=COLORS["text_secondary"],
            font=(FONTS["small"][0], int(FONTS["small"][1] * s), "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=padx, pady=(0, 8))

    def _card(self, parent: tk.Widget, padx: int) -> tk.Frame:
        card = tk.Frame(
            parent,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=2,
        )
        card.pack(fill=tk.X, padx=padx, pady=(0, int(16 * self._scale)))
        return card

    # -----------------------------------------------------------------------
    # Resize handling
    # -----------------------------------------------------------------------

    def _on_resize(self, event) -> None:
        new_scale = round(max(0.85, min(1.8, event.width / 960)), 2)
        if abs(new_scale - self._scale) > 0.05:
            self._scale = new_scale
            self._populate()

    def _on_canvas_resize(self, event) -> None:
        self._canvas.itemconfig(self._canvas_window, width=event.width)

    # -----------------------------------------------------------------------
    # Contact management
    # -----------------------------------------------------------------------

    def _add_contact_row(self) -> None:
        row = _ContactRow(
            self._contacts_container,
            on_remove=self._remove_contact_row,
            index=len(self._contact_rows),
            scale=self._scale,
        )
        row.pack(fill=tk.X, pady=(0, int(16 * self._scale)))
        self._contact_rows.append(row)

    def _remove_contact_row(self, row: _ContactRow) -> None:
        if len(self._contact_rows) <= 1:
            self._set_status(
                "At least one emergency contact is required.",
                color="warning",
            )
            return
        row.destroy()
        self._contact_rows.remove(row)
        for i, r in enumerate(self._contact_rows):
            r.index = i

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def _build_config(self, silent: bool = False) -> AlertConfig | None:
        user_name = self._user_name_var.get().strip()
        if not user_name or user_name == "e.g. Margaret Smith":
            if not silent:
                self._set_status(
                    "Please enter the name of the person being monitored.",
                    color="warning",
                )
            return None

        contacts = []
        for i, row in enumerate(self._contact_rows):
            name, phone = row.get()
            if not name or not phone:
                if not silent:
                    self._set_status(
                        f"Contact {i + 1} is incomplete — fill in both name and phone number.",
                        color="warning",
                    )
                return None
            if not phone.startswith("+"):
                if not silent:
                    self._set_status(
                        f"Contact {i + 1}: phone must include country code and start with + "
                        f"(e.g. +12125551234).",
                        color="warning",
                    )
                return None
            contacts.append(EmergencyContact(
                name=name,
                phone=phone,
                is_primary=(i == 0),
            ))

        return AlertConfig(user_name=user_name, contacts=contacts)

    # -----------------------------------------------------------------------
    # Button handlers
    # -----------------------------------------------------------------------

    def _on_test(self) -> None:
        config = self._build_config()
        if config is None:
            return

        self._set_status(
            "Starting test alert — the assistant will speak and listen for your response...",
            color="accent",
        )
        self.configure(cursor="watch")
        # Defer by 50ms so the status label paints before the blocking audio pipeline starts.
        # pyttsx3 and sounddevice both require the macOS main thread — do NOT use threading here.
        self.after(50, lambda: self._run_assessment(config))

    def _run_assessment(self, config) -> None:
        """Run the full assessment pipeline on the main thread (required for pyttsx3 + sounddevice on macOS)."""
        try:
            result = run_assessment(
                config=config,
                on_status=lambda msg: self._set_status(msg),
                test_mode=True,
            )
            if result.alert_sent:
                successes = sum(1 for r in result.alert_results if r.success)
                total     = len(result.alert_results)
                self._set_status(
                    f"✓  Test complete — {successes} of {total} actions succeeded.",
                    color="success",
                )
            else:
                self._set_status(
                    f"Check-in result: {result.outcome.value}. No alert was triggered.",
                    color="text_secondary",
                )
        except Exception as exc:
            self._set_status(f"Error: {exc}", color="danger")
        finally:
            self.configure(cursor="")

    def _on_save(self) -> None:
        config = self._build_config()
        if config is None:
            return
        try:
            import json
            data = {
                "user_name": config.user_name,
                "contacts": [
                    {"name": c.name, "phone": c.phone, "is_primary": c.is_primary}
                    for c in config.contacts
                ],
            }
            with open("config.json", "w") as f:
                json.dump(data, f, indent=2)
            self._set_status("✓  Configuration saved", color="success")
        except Exception as exc:
            self._set_status(f"Failed to save: {exc}", color="danger")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _set_status(self, msg: str, color: str = "text_secondary") -> None:
        def _update():
            self._status_var.set(msg)
            self._status_label.configure(fg=COLORS.get(color, COLORS["text_secondary"]))
        self.after(0, _update)

    def get_config(self) -> AlertConfig | None:
        """Called by monitoring screen when a fall is detected."""
        return self._build_config(silent=True)