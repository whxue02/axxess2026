# response/

The `response/` module handles everything that happens **after a fall is detected** — checking in with the user verbally, and dispatching emergency alerts if they need help or don't respond.

---

## Files

| File | Responsibility |
|---|---|
| `voice_assistant.py` | TTS check-in prompt + STT listening + timeout logic |
| `emergency_alert.py` | Twilio voice call to emergency contacts + mock 911 dispatch |
| `pipeline.py` | Combines the voice assistant and emergency alert into one function |
| `__init__.py` | Exposes the public interface of the module |

---

## How it works

```
Fall detected
     │
     ▼
VoiceAssistant.run_checkin()
     │
     ├── User says "I'm fine"  ──────────────────► Resume monitoring
     │
     ├── User says "help"
     │        or
     └── No response (timeout) ──────────────────► EmergencyAlerter.send_alert()
                                                         │
                                                         ├── Twilio call to all contacts
                                                         └── Mock 911 dispatch (console log)
```

---

## voice_assistant.py

### What it does

1. Speaks a check-in prompt aloud using **pyttsx3** (fully offline TTS)
2. Records microphone audio for up to `timeout_seconds`
3. Transcribes the audio using **pywhispercpp** (fully offline STT — no audio leaves the device)
4. Classifies the transcript as `SAFE`, `HELP_NEEDED`, or `NO_RESPONSE`
5. If there's no response, issues a second shorter prompt and listens again
6. Returns a `UserResponse` enum value to the caller

### Privacy

All audio is processed entirely on-device. No audio, transcript, or voice data is ever sent to an external server. Temporary `.wav` files are deleted immediately after transcription.

### Dependencies

```bash
pip install pyttsx3 sounddevice soundfile pywhispercpp
```

> **Python 3.10+ required.** pywhispercpp uses the `X | Y` union type syntax which is not supported in Python 3.9.

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `whisper_model` | `"base.en"` | Model name or path to a local ggml file. Downloaded automatically on first run. |
| `whisper_models_dir` | `None` | Directory to store/load model files. Uses pywhispercpp's default cache if not set. |
| `n_threads` | `4` | CPU threads for whisper inference. |
| `timeout_seconds` | `15` | How long to listen on the first prompt. |
| `second_chance_seconds` | `8` | How long to listen on the follow-up prompt (only reached on silence). |
| `tts_rate` | `145` | Speech rate in words per minute. |
| `tts_volume` | `1.0` | TTS volume (0.0 – 1.0). |

### Usage

```python
from response import VoiceAssistant, UserResponse

assistant = VoiceAssistant(
    whisper_model="base.en",
    timeout_seconds=15,
)

outcome = assistant.run_checkin()

if outcome == UserResponse.SAFE:
    pass  # resume monitoring
elif outcome in (UserResponse.HELP_NEEDED, UserResponse.NO_RESPONSE):
    pass  # trigger emergency alert
```

###  Testing

Run the file directly to test the full audio pipeline end to end:

```bash
python3 voice_assistant.py
```

The assistant will speak a prompt, record your response, transcribe it, and print the result.

### Model size tradeoff

| Model | Size | Transcription speed | Recommended for |
|---|---|---|---|
| `tiny.en` | ~75 MB | ~0.5s | Testing / fast hardware |
| `base.en` | ~142 MB | ~1–2s | **Production (recommended)** |

`base.en` is the right call for this use case — it handles slurred or partial speech significantly better than `tiny.en`, which matters when the user may be injured.

---

## emergency_alert.py

### What it does

1. Places an automated **Twilio voice call** to every configured emergency contact
2. Speaks a pre-written alert message twice (recipients may not process it on the first pass)
3. Logs a **mock emergency services dispatch** to console (replace with a real integration when ready)
4. Returns a list of `AlertResult` objects so the caller can see exactly what succeeded or failed

### Dependencies

```bash
pip install twilio python-dotenv
```

### Credentials — .env file

Create a `.env` file in your project root:

```
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_FROM_NUMBER=+18005550100

# For smoke testing only
TEST_CONTACT_PHONE=+12125551234
TEST_CONTACT_NAME=Susan
TEST_USER_NAME=Margaret
```

> All phone numbers must be in **E.164 format**: `+` followed by country code and number, no spaces or dashes (e.g. `+12125551234`).

Credentials are read automatically from `.env` when `EmergencyAlerter` is initialized. If any of the three Twilio variables are missing, an `EnvironmentError` is raised immediately with the names of the missing variables.

### Usage

```python
from response import EmergencyAlerter, AlertConfig, EmergencyContact

config = AlertConfig(
    user_name="Margaret",
    contacts=[
        EmergencyContact(name="Susan", phone="+12125551234", is_primary=True),
        EmergencyContact(name="David", phone="+13105559876"),
    ],
)

alerter = EmergencyAlerter(config)
results = alerter.send_alert()

for r in results:
    print(r.action, "—", "OK" if r.success else r.error)
```

### test_mode

Pass `test_mode=True` to send a clearly labeled test alert. Use this from the setup screen's "Send test alert" button to verify credentials and contact numbers without alarming anyone.

```python
alerter.send_alert(test_mode=True)
```

### Smoke test

```bash
python3 emergency_alert.py
```

Requires `TEST_CONTACT_PHONE` to be set in `.env`. Runs in `test_mode=True` automatically.

### AlertResult

Every action returns an `AlertResult`:

```python
@dataclass
class AlertResult:
    action: str       # e.g. "Call to Susan (+12125551234)"
    success: bool
    error: str | None  # None if successful
    timestamp: str     # ISO-8601
```

Each contact's call is attempted independently — a failed call to one contact does not prevent the others from being tried.

---

## Full pipeline test

`test_response_pipeline.py` (in the project root) runs the complete flow end to end:

```bash
python3 test_response_pipeline.py
```

Requires all `.env` variables to be set. Runs the voice check-in, then fires a `test_mode=True` alert if help is needed.

---

## Folder structure

```
response/
├── __init__.py           # Public interface
├── voice_assistant.py    # TTS check-in + STT listening + timeout logic
└── emergency_alert.py    # Twilio voice call + mock 911 dispatch
```