"""Pure functions behind the HTTP API (api.py). No web framework here, so the
whole payload logic is unit-testable without a server.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .data import load_accounts
from .insights import build_prompt, get_brief, segment_medians
from .scoring import OPP_FOCUS, RISK_FOCUS, score_accounts

TOP_N = 10
SIGNALS_TOTAL = 5  # four numeric rules + the call transcript label


@lru_cache(maxsize=1)
def scored() -> pd.DataFrame:
    return score_accounts(load_accounts())


def _one_liner(row: pd.Series, kind: str) -> str:
    reasons = row["risk_reasons"] if kind == "risk" else row["opportunity_reasons"]
    # Lower-case the first letter unless it starts an acronym ("AI feature…")
    bits = [r if (len(r) > 1 and r[1].isupper()) else r[0].lower() + r[1:] for r in reasons[:3]]
    text = "; ".join(bits).replace("last call: ", "last call mentioned ")
    return (text[0].upper() + text[1:]) if text else "No strong signals"


def _row(row: pd.Series, kind: str) -> dict:
    return {
        "account_id": row["account_id"],
        "name": row["account_name"],
        "segment": row["segment"],
        "region": row["region"],
        "industry": row["industry"],
        "revenue": float(row["current_revenue"]),
        "renews_in": int(row["days_to_next_renewal"]),
        "score": int(row["risk_score"] if kind == "risk" else row["opportunity_score"]),
        "signals": len(row["risk_reasons"] if kind == "risk" else row["opportunity_reasons"]),
        "signals_total": SIGNALS_TOTAL,
        "priority": float(row["priority_score"]),
        "why": _one_liner(row, kind),
        "has_brief": get_brief(row["account_id"]) is not None,
    }


def apply_filters(df: pd.DataFrame, segment: str | None, region: str | None, industry: str | None) -> pd.DataFrame:
    if segment:
        df = df[df["segment"] == segment]
    if region:
        df = df[df["region"] == region]
    if industry:
        df = df[df["industry"] == industry]
    return df


def focus_payload(segment: str | None = None, region: str | None = None, industry: str | None = None,
                  top_n: int = TOP_N) -> dict:
    """Everything the Today's-focus page needs in one call."""
    all_ = scored()
    book = apply_filters(all_, segment, region, industry)
    # Order by priority (score x urgency x value), per docs/decisions.md #6.
    # Sorting by raw score alone pins every capped-at-100 account to the top.
    save = book[book["focus"] == "risk"].sort_values(["priority_score", "risk_score"], ascending=False)
    grow = book[book["focus"] == "opportunity"].sort_values(["priority_score", "opportunity_score"], ascending=False)
    rest = book[~book["focus"].isin(["risk", "opportunity"])]
    return {
        "filters": {
            "segment": sorted(all_["segment"].unique()),
            "region": sorted(all_["region"].unique()),
            "industry": sorted(all_["industry"].unique()),
        },
        "kpis": {
            "book": int(len(book)),
            "save_count": int(len(save)),
            "save_revenue": float(save["current_revenue"].sum()),
            "grow_count": int(len(grow)),
            "grow_revenue": float(grow["current_revenue"].sum()),
            "renewals_45d": int(((book["focus"].isin(["risk", "opportunity"])) & (book["days_to_next_renewal"] <= 45)).sum()),
            "thresholds": {"risk": RISK_FOCUS, "opportunity": OPP_FOCUS},
        },
        "save": [_row(r, "risk") for _, r in save.head(top_n).iterrows()],
        "grow": [_row(r, "opportunity") for _, r in grow.head(top_n).iterrows()],
        "rest": [
            {"account_id": r["account_id"], "name": r["account_name"], "segment": r["segment"], "region": r["region"],
             "status": r["focus"], "risk": int(r["risk_score"]), "opportunity": int(r["opportunity_score"]),
             "renews_in": int(r["days_to_next_renewal"]), "revenue": float(r["current_revenue"])}
            for _, r in rest.sort_values("priority_score", ascending=False).iterrows()
        ],
    }


def account_payload(account_id: str) -> dict | None:
    """Everything the brief drawer needs for one account."""
    df = scored()
    if account_id not in set(df["account_id"]):
        return None
    acc = df.set_index("account_id").loc[account_id].to_dict() | {"account_id": account_id}
    peers = segment_medians(df, acc["segment"])
    prompt, card = build_prompt(acc, peers)
    cached = get_brief(account_id)
    return {
        "account_id": account_id,
        "name": acc["account_name"],
        "description": acc["account_description"],
        "industry": acc["industry"], "segment": acc["segment"], "region": acc["region"],
        "employees": int(acc["nr_employees"]),
        "revenue": float(acc["current_revenue"]),
        "renews_in": int(acc["days_to_next_renewal"]),
        "focus": acc["focus"],
        "action": {"risk": "Save", "opportunity": "Grow", "watch": "Watch", "healthy": "Healthy"}[acc["focus"]],
        "risk_score": int(acc["risk_score"]),
        "opportunity_score": int(acc["opportunity_score"]),
        "risk_reasons": list(acc["risk_reasons"]),
        "opportunity_reasons": list(acc["opportunity_reasons"]),
        "facts": {
            "Seats licensed / active": f"{acc['nr_licensed_seats']:,} / {acc['nr_active_users']:,} ({acc['seat_utilisation']:.0%} used)",
            "Days since last sales activity": f"{int(acc['days_since_last_sales_activity'])}"
                                              + (" (estimated)" if acc.get("days_since_last_sales_activity_missing") else ""),
            "Open support tickets": f"{int(acc['nr_support_tickets'])}",
            "AI feature adoption (0–1)": f"{acc['ai_usage']:.2f}" + (" (estimated)" if acc.get("ai_usage_missing") else ""),
            "Last call": acc["call_transcript_summary"] if isinstance(acc.get("call_transcript_summary"), str)
                         else "No call summary on file",
            "Call label": f"{acc['intent']} · {acc.get('risk_type', 'none')} · {acc.get('expansion_type', 'none')}",
        },
        "brief": cached["brief"] if cached else None,
        "brief_meta": {"generated_on": cached["generated_on"], "generator": cached["generator"].split(" (")[0]} if cached else None,
        "prompt": prompt,
        "context_card": card,
    }
