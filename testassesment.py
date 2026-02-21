from response.pipeline import run_assessment, AssessmentResult
from response import AlertConfig, EmergencyContact

config = AlertConfig(
    user_name="Margaret",
    contacts=[EmergencyContact("Susan", "+19132440654", is_primary=True)],
)

result = run_assessment(
    config=config,
    on_status=lambda msg: print(msg), 
)

print(result.outcome)        # "safe" | "help_needed" | "no_response"
print(result.alert_sent)     # True if emergency alert was fired
print(result.alert_results)  # list[AlertResult] — one per contact + mock dispatch