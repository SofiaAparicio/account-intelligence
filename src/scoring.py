"""Scoring layer: which accounts need attention, and why.

Two separate scores, because an AE acts differently on "save this" vs "grow this":

- risk_score (0-100):        renewal proximity amplified by low engagement,
                             open tickets, low AI usage, low seat utilisation
- opportunity_score (0-100): seat-cap pressure, high AI usage, expansion
                             language in the last call
- priority_score:            max(risk, opportunity) weighted by revenue at stake
                             (log-scaled so Enterprise does not swamp everything)

Every score carries human-readable reason codes so the ranking is explainable
in the UI and in the room.

Validation: validate_against_ground_truth() measures how much of the actual
churn / expansion (from revenue_end_of_quarter) lands in the top decile.
"""

# TODO(step 3): implement; weights and thresholds informed by the EDA
