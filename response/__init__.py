"""
response/__init__.py

Public interface for the response module.

Usage
-----
    from response import run_assessment, AssessmentResult
    from response import AlertConfig, EmergencyContact
    from response import VoiceAssistant, UserResponse
"""

from response.pipeline import run_assessment, AssessmentResult
from response.voice_assistant import VoiceAssistant, UserResponse
from response.emergency_alert import EmergencyAlerter, AlertConfig, EmergencyContact, AlertResult

__all__ = [
    # Primary entry point — this is what the frontend calls
    "run_assessment",
    "AssessmentResult",
    # Config types needed to call run_assessment
    "AlertConfig",
    "EmergencyContact",
    # Lower-level classes (available if needed directly)
    "VoiceAssistant",
    "UserResponse",
    "EmergencyAlerter",
    "AlertResult",
]