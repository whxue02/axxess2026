"""
TTS check-in + STT listening + timeout logic for the fall detection system.

Flow:
    1. Fall confirmed by detection pipeline
    2. VoiceAssistant.run_checkin() is called
    3. TTS speaks a check-in prompt (pyttsx3, fully offline)
    4. Mic audio is recorded for `timeout_seconds`
    5. Audio is transcribed via pywhispercpp (fully offline, no subprocess)
    6. Transcript is classified into UserResponse enum
    7. Only if response is NO_RESPONSE does a second prompt play
    8. Final UserResponse is returned to the caller

Dependencies:
    pip install pyttsx3 sounddevice soundfile pywhispercpp

    pywhispercpp will automatically download the requested model on first run,
    or you can point it at a local ggml model file via `model` parameter.
    See: https://github.com/absadiki/pywhispercpp
"""

from __future__ import annotations

import logging
import tempfile
import time
from enum import Enum
from pathlib import Path

import pyttsx3
import sounddevice as sd
import soundfile as sf
from pywhispercpp.model import Model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------

class UserResponse(Enum):
    """
    The three possible outcomes of a check-in session.

    SAFE         — user verbally confirmed they are okay; resume monitoring.
    HELP_NEEDED  — user verbally requested help; trigger emergency alert.
    NO_RESPONSE  — user did not respond within the timeout window (or audio
                   pipeline failed); default to emergency alert.
    """
    SAFE = "safe"
    HELP_NEEDED = "help_needed"
    NO_RESPONSE = "no_response"


# ---------------------------------------------------------------------------
# Keyword lists for response classification
# ---------------------------------------------------------------------------

# Words/phrases that indicate the user is safe.
# Kept broad to account for slurred or partial speech.
SAFE_KEYWORDS = [
    "fine", "okay", "ok", "good", "alright", "all right",
    "i'm fine", "im fine", "i am fine",
    "i'm okay", "im okay", "i am okay",
    "i'm good", "im good", "i am good",
    "no help", "don't call", "do not call", "not hurt",
]

# Words/phrases that indicate the user needs help.
HELP_KEYWORDS = [
    "help", "hurt", "pain", "fallen", "can't get up", "cannot get up",
    "call", "emergency", "ambulance", "yes", "please",
    "i need help", "need help", "i fell",
]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class VoiceAssistant:
    """
    Handles the verbal check-in sequence after a fall is detected.

    Parameters
    ----------
    whisper_model : str
        Model name (e.g. "base.en", "tiny.en") or a direct path to a local
        ggml model file. Named models are downloaded automatically on first use
        and cached locally by pywhispercpp. "base.en" is recommended — it is
        fast on CPU and accurate enough for short spoken responses.

    whisper_models_dir : str | None
        Optional directory where pywhispercpp stores/looks for model files.
        Defaults to pywhispercpp's own cache directory if None.

    n_threads : int
        Number of CPU threads for whisper inference. 4 is a safe default
        for most machines; increase on higher-core-count CPUs.

    timeout_seconds : int
        How long (in seconds) to listen for a response on the first prompt.
        Recommended: 15 seconds.

    second_chance_seconds : int
        How long to listen on the follow-up prompt if the first yields
        NO_RESPONSE (silence). SAFE and HELP_NEEDED on the first prompt
        resolve immediately without a second prompt.

    sample_rate : int
        Audio sample rate for recording. Whisper requires 16000 Hz.

    tts_rate : int
        Speech rate for pyttsx3 in words-per-minute.
        Lower values are clearer for elderly users. Default: 145.

    tts_volume : float
        TTS volume from 0.0 to 1.0. Default: 1.0 (maximum).
    """

    def __init__(
        self,
        whisper_model: str = "base.en",
        whisper_models_dir: str | None = None,
        n_threads: int = 4,
        timeout_seconds: int = 6,
        second_chance_seconds: int = 4,
        sample_rate: int = 16000,
        tts_rate: int = 145,
        tts_volume: float = 1.0,):

        
        self.timeout = timeout_seconds
        self.second_chance_timeout = second_chance_seconds
        self.sample_rate = sample_rate
        self._tts_rate = tts_rate
        self._tts_volume = tts_volume
        self._tts_voice_id: str | None = None  # set during _init_tts

        # --- Load whisper model via pywhispercpp ---
        # Loaded once and reused — avoids cold-start cost on each check-in.
        logger.info("Loading whisper model '%s' (this may take a moment)...", whisper_model)
        self._whisper = Model(
            model=whisper_model,
            models_dir=whisper_models_dir,
            print_realtime=False,
            print_progress=False,
            n_threads=n_threads,
        )
        logger.info("Whisper model loaded.")

        # Probe available voices once and store the preferred voice ID.
        # speak() re-initializes the engine each call (macOS pyttsx3 bug fix),
        # so we can't store the engine instance long-term.
        self._init_tts()

        logger.info(
            "VoiceAssistant initialized | model=%s | timeout=%ds",
            whisper_model,
            self.timeout,
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run_checkin(self) -> UserResponse:
        """
        Execute a full check-in sequence after a fall is detected.

        Decision tree:
            First prompt
            ├── SAFE        → speak confirmation, return SAFE
            ├── HELP_NEEDED → speak confirmation, return HELP_NEEDED (no second prompt)
            └── NO_RESPONSE → second prompt
                              ├── SAFE        → speak confirmation, return SAFE
                              ├── HELP_NEEDED → speak confirmation, return HELP_NEEDED
                              └── NO_RESPONSE → speak alert, return NO_RESPONSE

        Returns
        -------
        UserResponse
            Callers should treat HELP_NEEDED and NO_RESPONSE identically —
            both should trigger the emergency alert pipeline.
        """
        logger.info("Starting fall check-in sequence.")

        # --- First prompt ---
        self.speak(
            "I noticed you may have fallen. Are you okay? "
            "Please say 'I'm fine' if you are safe, "
            "or say 'help' if you need assistance."
        )

        response = self._listen_and_classify(duration=self.timeout)
        logger.info("First prompt response: %s", response)

        if response == UserResponse.SAFE:
            self.speak("Okay, I'm glad you're safe. I'll continue monitoring.")
            return UserResponse.SAFE

        if response == UserResponse.HELP_NEEDED:
            logger.info("Help requested on first prompt — skipping second prompt.")
            self.speak("Okay, I'm contacting your emergency contacts now. Help is on the way.")
            return UserResponse.HELP_NEEDED

        # --- Second chance prompt (only reached on silence / NO_RESPONSE) ---
        logger.info("No response on first prompt. Issuing second prompt.")
        self.speak(
            "I did not hear a response. "
            "Please say 'I'm fine' if you are okay, or say 'help' if you need assistance."
        )

        response = self._listen_and_classify(duration=self.second_chance_timeout)
        logger.info("Second prompt response: %s", response)

        if response == UserResponse.SAFE:
            self.speak("Okay, I'm glad you're safe. I'll continue monitoring.")
            return UserResponse.SAFE

        if response == UserResponse.HELP_NEEDED:
            self.speak("Okay, I'm contacting your emergency contacts now. Help is on the way.")
            return UserResponse.HELP_NEEDED

        # --- No response after two attempts ---
        logger.warning("No response after two prompts. Triggering emergency alert pipeline.")
        self.speak("I'm going to contact your emergency contacts now. Help is on the way.")
        return UserResponse.NO_RESPONSE

    def speak(self, message: str) -> None:
        """
        Speak a message aloud using pyttsx3.

        A fresh engine is created for each call. This works around a known
        macOS bug where pyttsx3 silently fails on runAndWait() calls after
        the first one when reusing the same engine instance.

        Parameters
        ----------
        message : str
            The text to speak.
        """
        logger.info("Speaking: %s", message)
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self._tts_rate)
            engine.setProperty("volume", self._tts_volume)
            if self._tts_voice_id:
                engine.setProperty("voice", self._tts_voice_id)
            engine.say(message)
            engine.runAndWait()
            engine.stop()
        except Exception as exc:
            logger.error("TTS failed: %s", exc, exc_info=True)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _init_tts(self) -> None:
        """
        Probe pyttsx3 once to find and store the preferred voice ID.
        The ID is reused in every speak() call without keeping an engine
        instance alive between calls.
        """
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            for voice in voices:
                if "female" in voice.name.lower() or "zira" in voice.name.lower():
                    self._tts_voice_id = voice.id
                    logger.debug("Preferred TTS voice: %s", voice.name)
                    break
            engine.stop()
        except Exception as exc:
            logger.debug("Could not probe TTS voices: %s", exc)

    def _listen_and_classify(self, duration: int) -> UserResponse:
        """
        Record audio for `duration` seconds, transcribe with pywhispercpp,
        and classify the result. Temp file is always cleaned up.

        Parameters
        ----------
        duration : int
            Recording duration in seconds.

        Returns
        -------
        UserResponse
        """
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name

            logger.info("Recording audio for %d seconds -> %s", duration, tmp_path)
            self._record_audio(duration=duration, filepath=tmp_path)

            transcript = self._transcribe(audio_path=tmp_path)
            logger.info("Whisper transcript: '%s'", transcript)

            return self._classify_response(transcript)

        except Exception as exc:
            # Any failure in the audio pipeline defaults to NO_RESPONSE,
            # which triggers the emergency alert — the safest fallback.
            logger.error("Error in listen/classify pipeline: %s", exc, exc_info=True)
            return UserResponse.NO_RESPONSE

        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    def _record_audio(self, duration: int, filepath: str) -> None:
        """
        Record mono 16kHz audio from the default input device.

        Parameters
        ----------
        duration : int
            Number of seconds to record.
        filepath : str
            Destination .wav file path.
        """
        audio = sd.rec(
            frames=int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
        )
        sd.wait()
        sf.write(filepath, audio, self.sample_rate)
        logger.debug("Audio saved to %s", filepath)

    def _transcribe(self, audio_path: str) -> str:
        """
        Transcribe an audio file using pywhispercpp (in-process, fully offline).

        Parameters
        ----------
        audio_path : str
            Path to the .wav file to transcribe.

        Returns
        -------
        str
            Lowercased transcript text, or empty string on failure.
        """
        segments = self._whisper.transcribe(audio_path)
        transcript = " ".join(seg.text for seg in segments).strip().lower()
        return transcript

    def _classify_response(self, transcript: str) -> UserResponse:
        """
        Classify a transcript string into a UserResponse value.

        SAFE is checked before HELP_NEEDED so that a phrase like
        "I'm fine, no help needed" doesn't accidentally match "help".

        Parameters
        ----------
        transcript : str
            Lowercased transcript from whisper.

        Returns
        -------
        UserResponse
        """
        if not transcript or transcript == "[blank_audio]":
            return UserResponse.NO_RESPONSE

        if any(keyword in transcript for keyword in SAFE_KEYWORDS):
            return UserResponse.SAFE

        if any(keyword in transcript for keyword in HELP_KEYWORDS):
            return UserResponse.HELP_NEEDED

        return UserResponse.NO_RESPONSE



# test
if __name__ == "__main__":
    print("Loading VoiceAssistant with model 'base.en'...")
    print("Starting test check-in in 3 seconds...")
    time.sleep(3)

    assistant = VoiceAssistant(
        whisper_model="base.en",
        timeout_seconds=12,
        second_chance_seconds=7,
    )

    outcome = assistant.run_checkin()
    print(f"\nCheck-in result: {outcome.value}")

    if outcome in (UserResponse.HELP_NEEDED, UserResponse.NO_RESPONSE):
        print("-> Emergency alert pipeline would be triggered.")
    else:
        print("-> User is safe. Returning to monitoring.")