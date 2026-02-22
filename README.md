# Fall Detection System

A real-time fall detection system for elderly users living independently. The system monitors a person via webcam, detects falls using pose estimation and a trained classifier, checks in verbally after a confirmed fall, and automatically calls emergency contacts if help is needed or there is no response.

All video and audio processing happens entirely on-device. No camera feed, audio, or personal data is ever sent to an external server.

---

## How it works

```
Webcam feed
     │
     ▼
DetectionPipeline (MediaPipe pose + Random Forest classifier)
     │
     ├── near_fall ──────────────────────────────► Log to Event Log
     │
     └── fall (confirmed) ───────────────────────► Log to Event Log
                                                    Save fall clip (skeleton only)
                                                         │
                                                         ▼
                                              VoiceAssistant check-in
                                              (speaks prompt, listens for response)
                                                         │
                                                         ├── "I'm fine" ──► Resume monitoring
                                                         │
                                                         └── "Help" / no response
                                                                  │
                                                                  ▼
                                                     EmergencyAlerter
                                                     (Twilio voice call to all contacts)
```

---

## Requirements

- Python 3.10+
- macOS or Linux (tested on macOS — pyttsx3 uses the system `nsss` speech driver)
- Webcam

---

## Installation

**1. Clone the repo and create a virtual environment**

```bash
git clone https://github.com/your-org/axxess2026.git
cd axxess2026
python3.10 -m venv venv
source venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, install the packages directly:

```bash
pip install opencv-python mediapipe numpy scikit-learn \
            pyttsx3 sounddevice soundfile pywhispercpp \
            twilio python-dotenv pillow
```

> On Apple Silicon, install pywhispercpp from source for Metal/CoreML acceleration:
> ```bash
> pip install git+https://github.com/absadiki/pywhispercpp
> ```

**3. Create your `.env` file**

```
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_FROM_NUMBER=+18005550100
```

All phone numbers must be in E.164 format (e.g. `+12125551234`).

---

## Running the app

```bash
python3.10 main.py
```

The app opens on the **Setup** screen. Fill in the monitored user's name and at least one emergency contact, then click **Save Configuration**. After that, navigate to the **Monitor** tab and click **▶ Start Monitoring**.

---

## Screens

### Setup
Configure the monitored user's name and emergency contacts. Click **Send Test Alert** to place a test Twilio call and verify your credentials and contact numbers are correct before going live.

Configuration is saved to `config.json` and loaded automatically the next time the app starts.

### Monitor
Shows a live skeleton overlay (no raw video — the person's image is never displayed or stored). A colour-coded status badge shows the current detection state:

| Badge | Meaning |
|---|---|
| ● MONITORING | Running normally, no event |
| ◌ CONFIRMING FALL… | Classifier is building confidence |
| ⚠ NEAR FALL DETECTED | Near-fall rules fired |
| ⚠ FALL DETECTED | Fall confirmed — assessment starting |
| ◉ RUNNING ASSESSMENT | Voice check-in in progress |

When a fall is confirmed, the app speaks a check-in prompt and listens for a response. If the user says they are fine, monitoring resumes. If they ask for help or don't respond, Twilio calls all configured emergency contacts. A post-alert overlay stays on screen until manually dismissed.

### Event Log
A chronological log of all falls, near-falls, assessment outcomes, and system messages. Newest events appear at the top. Fall entries include a **▶ Play fall clip** button that opens the saved clip in your system video player.

The log persists across sessions in `event_log.json` and is loaded automatically on startup.

---

## Privacy

- The live camera feed is **never shown** in the UI — only a skeleton overlay is displayed
- Fall clips are saved as skeleton-only video — the person's image is never recorded
- All speech recognition runs on-device via **Whisper** (pywhispercpp) — no audio leaves the device
- No video, audio, or personal data is sent to any external server
- The only outbound network calls are Twilio voice calls when an alert is triggered

---

## Configuration files

| File | Created by | Purpose |
|---|---|---|
| `.env` | You (manually) | Twilio credentials |
| `config.json` | Setup screen Save button | User name + emergency contacts |
| `event_log.json` | App on close | Persisted event history |
| `storage/fall_clips/` | Detection pipeline | Skeleton-only fall clip videos |


---

## Troubleshooting

**App crashes with GIL error during assessment**
Do not call `self.after()` from background threads. All Tkinter calls must happen on the main thread. The capture loop uses a `queue.Queue` to pass data to the main thread safely.

**TTS speaks but nothing is heard / assessment doesn't run**
`pyttsx3` and `sounddevice` require the macOS main thread. Do not run `run_assessment()` on a background thread.

**Whisper model download on first run**
The `base.en` model (~142 MB) is downloaded automatically on first run by pywhispercpp. This only happens once.

**`event_log.json` not appearing in expected location**
The file is saved to the project root (resolved from `ui/app.py`'s location) regardless of the working directory when the app is launched.

**Twilio call not going through**
Check that `TWILIO_FROM_NUMBER` is a verified Twilio number and that the destination number is verified in your Twilio account (required for trial accounts).
