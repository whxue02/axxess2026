"""
response/pipeline.py

Single entry point for the post-fall response sequence.

The frontend (or any other caller) only needs to call run_assessment().
All routing logic — check-in → classify → alert — lives here so that
nothing outside this module needs to know about UserResponse or AlertResult.

Usage
-----
    from response.pipeline import run_assessment, AssessmentResult
    from response import AlertConfig, EmergencyContact

    config = AlertConfig(
        user_name="Margaret",
        contacts=[EmergencyContact("Susan", "+12125551234", is_primary=True)],
    )

    result = run_assessment(
        config=config,
        on_status=lambda msg: print(msg),  # hook this to your UI
    )

    print(result.outcome)        # "safe" | "help_needed" | "no_response"
    print(result.alert_sent)     # True if emergency alert was fired
    print(result.alert_results)  # list[AlertResult] — one per contact + mock dispatch
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from response.voice_assistant import VoiceAssistant, UserResponse
from response.emergency_alert import EmergencyAlerter, AlertConfig, AlertResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class AssessmentResult:
    """
    The complete outcome of a single fall assessment run.

    Attributes
    ----------
    outcome : UserResponse
        The classified response from the voice check-in.
    alert_sent : bool
        True if the emergency alert sequence was triggered.
    alert_results : list[AlertResult]
        One entry per alert action (call per contact + mock dispatch).
        Empty if alert_sent is False.
    timestamp : str
        ISO-8601 timestamp of when the assessment started.
    """
    outcome: UserResponse
    alert_sent: bool = False
    alert_results: list[AlertResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_assessment(
    config: AlertConfig,
    on_status: Callable[[str], None] | None = None,
    voice_assistant: VoiceAssistant | None = None,
    test_mode: bool = False,
) -> AssessmentResult:
    """
    Run the full post-fall response sequence.

    Steps:
        1. Speak a check-in prompt and listen for the user's response
        2. If the user is safe, return immediately — no alert sent
        3. If help is needed or there is no response, fire the emergency alert

    Parameters
    ----------
    config : AlertConfig
        User name and emergency contacts. Twilio credentials are read
        from the .env file automatically by EmergencyAlerter.

    on_status : Callable[[str], None] | None
        Optional callback invoked at each stage with a human-readable
        status string. Hook this to your frontend to update the UI in
        real time (e.g. set a status label, log to the event panel).
        If None, status messages are only written to the logger.

    voice_assistant : VoiceAssistant | None
        Optional pre-initialized VoiceAssistant instance. If None, one
        is created with default settings. Pass your own to control model
        choice, timeouts, etc., or to inject a mock in tests.

    test_mode : bool
        If True, the emergency alert is sent with test_mode=True (calls
        are labeled as tests). Useful for the setup screen's test button.

    Returns
    -------
    AssessmentResult
        Full record of what happened — outcome, whether an alert was sent,
        and the result of each individual alert action.
    """
    timestamp = datetime.now().isoformat()
    logger.info("Assessment started | user=%s | test=%s", config.user_name, test_mode)

    def status(msg: str) -> None:
        """Emit a status update to the callback and the logger."""
        logger.info("[STATUS] %s", msg)
        if on_status:
            on_status(msg)

    # --- Step 1: Voice check-in ---
    status("Checking in with user...")

    assistant = voice_assistant or VoiceAssistant()
    outcome = assistant.run_checkin()

    logger.info("Check-in outcome: %s", outcome)

    # --- Step 2: User is safe — nothing more to do ---
    if outcome == UserResponse.SAFE:
        status("User confirmed safe. Resuming monitoring.")
        return AssessmentResult(
            outcome=outcome,
            alert_sent=False,
            timestamp=timestamp,
        )

    # --- Step 3: Help needed or no response — fire the alert ---
    reason = (
        "User requested help."
        if outcome == UserResponse.HELP_NEEDED
        else "No response received."
    )
    status(f"{reason} Contacting emergency contacts...")

    try:
        alerter = EmergencyAlerter(config)
        alert_results = alerter.send_alert(test_mode=test_mode)

        successes = sum(1 for r in alert_results if r.success)
        total     = len(alert_results)
        status(f"Alert sequence complete — {successes}/{total} actions succeeded.")

        logger.info(
            "Assessment complete | outcome=%s | alerts=%d/%d succeeded",
            outcome.value,
            successes,
            total,
        )

        return AssessmentResult(
            outcome=outcome,
            alert_sent=True,
            alert_results=alert_results,
            timestamp=timestamp,
        )

    except EnvironmentError as exc:
        # Missing Twilio credentials — surface this clearly so the frontend
        # can show a meaningful error rather than a silent failure.
        msg = f"Alert failed — missing credentials: {exc}"
        status(msg)
        logger.error(msg)

        return AssessmentResult(
            outcome=outcome,
            alert_sent=False,
            alert_results=[
                AlertResult(
                    action="Emergency alert",
                    success=False,
                    error=str(exc),
                )
            ],
            timestamp=timestamp,
        )

    except Exception as exc:
        msg = f"Alert failed — unexpected error: {exc}"
        status(msg)
        logger.error(msg, exc_info=True)

        return AssessmentResult(
            outcome=outcome,
            alert_sent=False,
            alert_results=[
                AlertResult(
                    action="Emergency alert",
                    success=False,
                    error=str(exc),
                )
            ],
            timestamp=timestamp,
        )