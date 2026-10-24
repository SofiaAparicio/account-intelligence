"""Prompt templates for the insight layer.

Structure of the meeting-prep prompt
------------------------------------
1. Role:        senior sales strategist preparing an AE for one meeting
2. Context card: structured facts + derived features + both scores + reason
                 codes + last call summary + segment-peer comparison
3. Rules:       use only the numbers given; say what is unknown; no invented
                 names, dates or figures
4. Output shape: Situation / Risks / Opportunities / 3 talking points /
                 Recommended next action
"""

# TODO(step 4): write SYSTEM_PROMPT and build_context_card(account_row, peers)
