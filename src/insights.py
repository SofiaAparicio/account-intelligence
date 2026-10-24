"""Insight layer: generate and cache the AI meeting brief for an account.

- generate_brief(account_id) -> str
- Provider is chosen from whichever key is present in .env (see .env.example)
- Results are cached to data/cache/<account_id>.json so the demo never
  depends on a live API call; the UI exposes a "regenerate" button.
"""

# TODO(step 4): implement provider adapters (Groq / Gemini / Anthropic / Ollama)
