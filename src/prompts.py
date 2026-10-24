"""Prompt templates for the intelligence layer.

Two prompts live here:

- LABELER_PROMPT   turns a call summary into typed labels (used once, offline;
                   judgments are committed in data/labels/template_labels.json)
- BRIEF_PROMPT     meeting-prep brief for one account (see build_context_card)
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# 1. Labeler
# --------------------------------------------------------------------------
LABELER_PROMPT = """You are labelling summaries of the most recent sales call with a B2B SaaS
customer. For each summary return ONE JSON object with exactly these keys:

  intent          one of: "risk" | "growth" | "neutral"
  risk_type       one of: "budget" | "competitor" | "champion_left" |
                  "support_escalation" | "roi_doubt" | "dissatisfaction" |
                  "consolidation" | "churn_intent" | "none"
  expansion_type  one of: "seats" | "new_team" | "module" | "none"
  confidence      number 0-1: how sure you are about `intent`
  rationale       one short sentence quoting the words that drove the label

Rules
- "risk" means the summary contains a concrete signal that revenue may shrink
  or the account may leave: a named competitor, a budget cut we are in scope
  for, a departed champion, an unresolved escalation, doubts about ROI or
  adoption, user dissatisfaction, vendor consolidation against us, or a
  request to terminate.
- "growth" means the summary contains a concrete signal that revenue may grow:
  a seat request or proposal, a roll-out to another team, interest in an
  additional module, or a consolidation in our favour.
- "neutral" means neither. A flat budget is neutral. "Will re-evaluate at
  renewal" is neutral (normal procurement language). Interest deferred to a
  later fiscal year is neutral now; mention it in the rationale.
- Do not infer risk from a lack of enthusiasm alone ("not a strategic
  priority", "focused on other things"); that is neutral with low confidence.
- risk_type must be "none" unless intent is "risk"; expansion_type must be
  "none" unless intent is "growth". If intent is "growth" but the ask is not
  specific, use expansion_type "none" and lower confidence.
- Judge only the text. Do not use anything you know about the company name.

Summary:
{transcript}
"""

# --------------------------------------------------------------------------
# 2. Brief
# --------------------------------------------------------------------------
BRIEF_SCHEMA = {
    "situation": "2 sentences: who they are, where the relationship stands",
    "risks": "0-3 bullets, each citing the number or call signal behind it",
    "opportunities": "0-3 bullets, each citing the number or call signal behind it",
    "talking_points": "exactly 3, phrased as things the AE can say or ask",
    "next_action": "one line: the single best next step and when",
    "unknowns": "1-3 things the data cannot tell us that the AE should find out",
}

BRIEF_PROMPT = """You are a senior sales strategist preparing an Account Executive for a
meeting with one customer. You will be given a context card with everything
we know about the account. Write a meeting-prep brief.

Rules
- Use ONLY facts from the context card. Every risk and opportunity must name
  the number or call signal it rests on (e.g. "no sales touch in 134 days").
- Never invent names, dates, quotes, prices or product details.
- If something important is unknown, put it in `unknowns` instead of guessing.
- Peer comparisons use the percentiles given ("bottom 10% of Mid-Market for
  seat utilisation"); do not compute new statistics.
- Plain language an AE can read in 60 seconds. No marketing tone.

Return ONE JSON object with exactly these keys and nothing else:
{schema}

Context card:
{context_card}
"""


def build_context_card(acc: dict, peers: dict | None = None) -> str:
    """Render one scored account row (as dict) into the context card string.

    `acc` is a row from src.scoring.score_accounts(); `peers` optionally holds
    segment-level medians for comparison. Kept as plain text so the same card
    can be shown in the UI next to the brief.
    """
    lines = [
        f"Account: {acc['account_name']} ({acc['account_id']})",
        f"What they do: {acc['account_description']}",
        f"Industry: {acc['industry']} | Segment: {acc['segment']} | Region: {acc['region']} | "
        f"Employees: {acc['nr_employees']:,}",
        "",
        "Commercials",
        f"- Current annual revenue: ${acc['current_revenue']:,.0f} "
        f"(P{int(acc['current_revenue_pct'] * 100)} within {acc['segment']})",
        f"- Days to renewal: {acc['days_to_next_renewal']}",
        f"- Licensed seats: {acc['nr_licensed_seats']:,} | Active users: {acc['nr_active_users']:,} | "
        f"Seat utilisation: {acc['seat_utilisation']:.0%} (P{int(acc['seat_utilisation_pct'] * 100)} within segment)",
        "",
        "Engagement and health",
        f"- Days since last sales activity: {acc['days_since_last_sales_activity']} "
        f"(P{int(acc['days_since_last_sales_activity_pct'] * 100)} within segment; higher = longer silence)"
        + (" [imputed: value was missing]" if acc.get("days_since_last_sales_activity_missing") else ""),
        f"- Open support tickets: {acc['nr_support_tickets']} (P{int(acc['nr_support_tickets_pct'] * 100)})",
        f"- AI feature adoption: {acc['ai_usage']:.2f} (P{int(acc['ai_usage_pct'] * 100)})"
        + (" [imputed: value was missing]" if acc.get("ai_usage_missing") else ""),
        "",
        "Last sales call",
        f"- Summary: {acc['call_transcript_summary'] if isinstance(acc.get('call_transcript_summary'), str) else 'NO CALL SUMMARY ON FILE'}",
        f"- Labelled intent: {acc['intent']} | risk type: {acc.get('risk_type', 'none')} | "
        f"expansion type: {acc.get('expansion_type', 'none')}",
        "",
        "Scores (rule-based, see README)",
        f"- Risk {acc['risk_score']}/100: " + ("; ".join(acc['risk_reasons']) or "no risk signals"),
        f"- Opportunity {acc['opportunity_score']}/100: " + ("; ".join(acc['opportunity_reasons']) or "no growth signals"),
        f"- Focus: {acc['focus']} | Priority {acc['priority_score']} (urgency x{acc['urgency']})",
    ]
    if peers:
        lines += ["", f"Segment medians ({acc['segment']})"] + [f"- {k}: {v}" for k, v in peers.items()]
    return "\n".join(lines)
