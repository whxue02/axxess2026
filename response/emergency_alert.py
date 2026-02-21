# emergency alerting for the fall detection system.

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime

from dotenv import load_dotenv
from twilio.rest import Client as TwilioClient
from twilio.base.exceptions import TwilioRestException

# Logging
logger = logging.getLogger(__name__)


# Data classes
@dataclass
class EmergencyContact:
    name: str # display name used in log messages and the spoken call message.
    phone: str # phone number in E.164 format (e.g. "+12125551234").
    is_primary: bool = False # useful for the UI to indicate who is listed first.


@dataclass
class AlertConfig:
    user_name: str # the monitored user's name, spoken aloud in the call message.
    contacts: list[EmergencyContact] # all contacts to call


@dataclass
class AlertResult:
    action: str # e.g. "Call to Susan (+12125551234)".
    success: bool # whether the action completed without error.
    error: str | None = None # error message if success is False, otherwise None.
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat()) #timestamp of when the arlet was aattempted

# Main class
class EmergencyAlerter:
    """
    Places emergency calls after a fall is confirmed.

    Usage
    -----
        config = AlertConfig(
            user_name="Margaret",
            contacts=[
                EmergencyContact("Susan", "+12125551234", is_primary=True),
                EmergencyContact("David", "+13105559876"),
            ],
        )
        alerter = EmergencyAlerter(config)
        results = alerter.send_alert()

    Parameters
    ----------
    config : AlertConfig
        User name and contact list.
    dotenv_path : str | None
        Optional explicit path to your .env file.
        If None, python-dotenv searches upward from the current directory.
    """

    def __init__(self, config: AlertConfig, dotenv_path: str | None = None):
        self.config = config

        # Load .env file
        load_dotenv(dotenv_path=dotenv_path)

        sid   = os.environ.get("TWILIO_ACCOUNT_SID", "")
        token = os.environ.get("TWILIO_AUTH_TOKEN", "")
        self._from_number = os.environ.get("TWILIO_FROM_NUMBER", "")

        # Validate credentials before doing anything else
        missing = [
            name for name, val in [
                ("TWILIO_ACCOUNT_SID",  sid),
                ("TWILIO_AUTH_TOKEN",   token),
                ("TWILIO_FROM_NUMBER",  self._from_number),
            ]
            if not val
        ]
        if missing:
            raise EnvironmentError(
                f"Missing required .env variable(s): {', '.join(missing)}"
            )

        logger.info("Twilio credentials loaded | SID=%s...", sid[:5])
        self._twilio = TwilioClient(sid, token)


    # Public API
    def send_alert(self, test_mode: bool = False) -> list[AlertResult]:
        """
        Execute the full alert sequence:
            1. Call every emergency contact via Twilio.
            2. Mock-dispatch to emergency services (console log only).

        Each call is attempted independently — a failure on one contact
        does not prevent the others from being tried.

        Parameters
        ----------
        test_mode : bool
            If True, the spoken message is clearly labeled as a test.
            Use this from the setup screen's "Send test alert" button.

        Returns
        -------
        list[AlertResult]
            One entry per action attempted (N calls + 1 mock dispatch).
        """
        results: list[AlertResult] = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        logger.warning(
            "ALERT TRIGGERED | user=%s | test=%s | time=%s",
            self.config.user_name,
            test_mode,
            timestamp,
        )

        # --- 1. Call all contacts ---
        for contact in self.config.contacts:
            result = self._make_call(contact, test_mode)
            results.append(result)

        # --- 2. Mock emergency services dispatch ---
        result = self._mock_dispatch_emergency_services(timestamp, test_mode)
        results.append(result)

        successes = sum(1 for r in results if r.success)
        logger.info(
            "Alert sequence complete: %d/%d actions succeeded.",
            successes,
            len(results),
        )

        return results

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _make_call(
        self,
        contact: EmergencyContact,
        test_mode: bool,
    ) -> AlertResult:
        """
        Place a single Twilio voice call to one emergency contact.

        The message is delivered as TwiML passed directly to the API —
        no external URL required.

        Parameters
        ----------
        contact : EmergencyContact
            The recipient.
        test_mode : bool
            Speaks a test label at the start of the message if True.

        Returns
        -------
        AlertResult
        """
        action = f"Call to {contact.name} ({contact.phone})"
        twiml  = self._build_twiml(contact.name, test_mode)

        try:
            call = self._twilio.calls.create(
                twiml=twiml,
                from_=self._from_number,
                to=contact.phone,
            )
            logger.info(
                "Call placed | to=%s | sid=%s",
                contact.phone,
                call.sid,
            )
            print(f"✅ Call placed to {contact.name} ({contact.phone}) | SID: {call.sid}")
            return AlertResult(action=action, success=True)

        except TwilioRestException as exc:
            logger.error("Call failed | to=%s | error=%s", contact.phone, exc.msg)
            print(f"❌ Call failed to {contact.name} ({contact.phone}): {exc.msg}")
            return AlertResult(action=action, success=False, error=exc.msg)

        except Exception as exc:
            logger.error(
                "Call unexpected error | to=%s | error=%s",
                contact.phone,
                str(exc),
                exc_info=True,
            )
            print(f"❌ Unexpected error calling {contact.name} ({contact.phone}): {exc}")
            return AlertResult(action=action, success=False, error=str(exc))

    def _mock_dispatch_emergency_services(
        self,
        timestamp: str,
        test_mode: bool,
    ) -> AlertResult:
        """
        Simulate a dispatch to emergency services (console log only).

        Replace the body of this method with a real integration when ready.

        Parameters
        ----------
        timestamp : str
            Human-readable timestamp for the log entry.
        test_mode : bool
            Flags the dispatch as a test in the output.

        Returns
        -------
        AlertResult
        """
        action = "Mock dispatch to emergency services"
        prefix = "[TEST] " if test_mode else ""

        print(
            f"\n"
            f"{'=' * 60}\n"
            f"{prefix}EMERGENCY SERVICES DISPATCH (MOCK)\n"
            f"{'=' * 60}\n"
            f"  Time     : {timestamp}\n"
            f"  Subject  : {self.config.user_name}\n"
            f"  Incident : Possible fall — no response to automated check-in\n"
            f"  Action   : Dispatch to subject's address\n"
            f"  Status   : *** THIS IS A MOCK — no real dispatch was made ***\n"
            f"{'=' * 60}\n"
        )

        logger.info(
            "Mock emergency services dispatch logged | user=%s",
            self.config.user_name,
        )
        return AlertResult(action=action, success=True)

    def _build_twiml(self, contact_name: str, test_mode: bool) -> str:
        """
        Build the TwiML string spoken during the call.

        The message repeats once — contacts may not process the first
        pass before they realize what they're hearing.

        Parameters
        ----------
        contact_name : str
            Recipient's name, used in the salutation.
        test_mode : bool
            Prepends a test label to the message if True.

        Returns
        -------
        str
            A TwiML <Response> string passed directly to Twilio's API.
        """
        test_prefix = "This is a test of the emergency alert system. " if test_mode else ""

        message = (
            f"{test_prefix}"
            f"Hello {contact_name}. "
            f"This is an automated alert. "
            f"{self.config.user_name} may have fallen and did not respond "
            f"to a check-in. Please check on them immediately and call "
            f"emergency services if needed."
        )

        return (
            f'<Response>'
            f'<Say voice="alice">{message}</Say>'
            f'<Pause length="1"/>'
            f'<Say voice="alice">{message}</Say>'
            f'</Response>'
        )


# test
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load .env to read the test contact phone number for the smoke test
    load_dotenv()
    test_phone = os.environ.get("TEST_CONTACT_PHONE", "")
    test_name  = os.environ.get("TEST_CONTACT_NAME", "Test Contact")

    if not test_phone:
        print("❌ ERROR: TEST_CONTACT_PHONE not set in .env")
    else:
        config = AlertConfig(
            user_name="Margaret",
            contacts=[
                EmergencyContact(name=test_name, phone=test_phone, is_primary=True),
            ],
        )

        alerter = EmergencyAlerter(config)
        results = alerter.send_alert(test_mode=True)

        print("\n--- Alert Results ---")
        for r in results:
            status = "OK" if r.success else f"FAILED: {r.error}"
            print(f"  [{status}] {r.action}")