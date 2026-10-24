"""Data layer: load, clean and enrich the account dataset.

Public API
----------
load_accounts(path=RAW_PATH) -> DataFrame   enriched accounts, no future columns
load_ground_truth(path=RAW_PATH) -> DataFrame   account_id, revenue_end_of_quarter, outcome
tag_transcript(text) -> str                 keyword fallback for the LLM labeler

Known data traps (see notebooks/01_eda.ipynb and docs/decisions.md)
------------------------------------------------------------------
- region "NA" is North America. pandas' default NA parsing turns it into NaN
  and silently drops 40% of accounts. Always read with keep_default_na=False.
- revenue_end_of_quarter is a *future* outcome. It never leaves
  load_ground_truth().
- ai_usage (30), days_since_last_sales_activity (20) and
  call_transcript_summary (52) have real missing values.
- 33 SMB accounts have more licensed seats than employees.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "account_data.csv"
LABELS_PATH = ROOT / "data" / "labels" / "transcript_labels.csv"

FUTURE_COLUMNS = ["revenue_end_of_quarter"]
NUMERIC_WITH_BLANKS = ["days_since_last_sales_activity", "ai_usage"]
PEER_METRICS = [
    "seat_utilisation",
    "ai_usage",
    "days_since_last_sales_activity",
    "nr_support_tickets",
    "current_revenue",
]

# --------------------------------------------------------------------------
# Keyword tagger: the no-API fallback for the LLM labeler (EDA section 7)
# --------------------------------------------------------------------------
RISK_TERMS = [
    "budget cut", "cost reduction", "spend reduction", "competitor", "ran a demo",
    "not renew", "may not renew", "disappoint", "cancel", "escalation", "unresolved",
    "in-house", "has left", "new leadership", "reviewing all vendor", "merging",
    "consolidation", "concerns about roi", "limited adoption", "threatened",
]
GROWTH_TERMS = [
    "expansion", "expand", "roll out", "rolling out", "additional seats", "seat add-on",
    "extend to", "case study", "more teams", "entire", "requesting",
]


def tag_transcript(text: object) -> str:
    """Return 'risk', 'growth', 'neutral' or 'none' (no transcript)."""
    if not isinstance(text, str) or not text.strip():
        return "none"
    s = text.lower()
    if any(k in s for k in RISK_TERMS):
        return "risk"
    if any(k in s for k in GROWTH_TERMS):
        return "growth"
    return "neutral"


# --------------------------------------------------------------------------
# Load / clean / enrich
# --------------------------------------------------------------------------
def load_raw(path: Path | str = RAW_PATH) -> pd.DataFrame:
    """Read the CSV without losing North America."""
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    for col in NUMERIC_WITH_BLANKS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["call_transcript_summary"] = df["call_transcript_summary"].replace("", np.nan)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Flag anomalies and impute, keeping a flag for every imputed value."""
    out = df.copy()

    out["no_recent_call"] = out["call_transcript_summary"].isna()
    out["seats_exceed_employees"] = out["nr_licensed_seats"] > out["nr_employees"]

    for col in NUMERIC_WITH_BLANKS:
        flag = f"{col}_missing"
        out[flag] = out[col].isna()
        medians = out.groupby("segment")[col].transform("median")
        out[col] = out[col].fillna(medians)

    out["days_since_last_sales_activity"] = out["days_since_last_sales_activity"].astype(int)
    return out


def add_features(df: pd.DataFrame, labels: pd.DataFrame | None = None) -> pd.DataFrame:
    """Derived columns used by scoring and by the LLM context card."""
    out = df.copy()

    out["seat_utilisation"] = out["nr_active_users"] / out["nr_licensed_seats"]
    out["revenue_per_seat"] = out["current_revenue"] / out["nr_licensed_seats"]
    out["seat_headroom"] = np.where(
        out["seats_exceed_employees"], np.nan, out["nr_employees"] - out["nr_licensed_seats"]
    )
    out["transcript_tag"] = out["call_transcript_summary"].map(tag_transcript)

    # LLM labels (typed) override the keyword tag when available
    if labels is not None and len(labels):
        out = out.merge(labels, on="account_id", how="left")
        out["intent"] = out["intent"].fillna(out["transcript_tag"])
    else:
        out["intent"] = out["transcript_tag"]
        out["risk_type"] = np.where(out["intent"] == "risk", "unspecified", "none")
        out["expansion_type"] = np.where(out["intent"] == "growth", "unspecified", "none")
        out["label_source"] = "keyword"

    # Segment-peer percentiles: "where does this account sit among companies like it?"
    for col in PEER_METRICS:
        out[f"{col}_pct"] = (
            out.groupby("segment")[col].rank(pct=True).round(2)
        )
    return out


def load_labels(path: Path | str = LABELS_PATH) -> pd.DataFrame | None:
    """Committed LLM labels, if the labeler has been run."""
    path = Path(path)
    if not path.exists():
        return None
    labels = pd.read_csv(path, keep_default_na=False)
    labels["label_source"] = "llm"
    return labels[["account_id", "intent", "risk_type", "expansion_type", "label_source"]]


def load_accounts(path: Path | str = RAW_PATH, with_labels: bool = True) -> pd.DataFrame:
    """The one function the app calls. Never contains future columns."""
    df = load_raw(path).drop(columns=FUTURE_COLUMNS)
    df = clean(df)
    labels = load_labels() if with_labels else None
    return add_features(df, labels)


def load_ground_truth(path: Path | str = RAW_PATH) -> pd.DataFrame:
    """Outcome labels derived from revenue_end_of_quarter. Validation ONLY."""
    df = load_raw(path)[["account_id", "current_revenue", "revenue_end_of_quarter"]].copy()
    delta = df["revenue_end_of_quarter"] - df["current_revenue"]
    df["outcome"] = np.select(
        [df["revenue_end_of_quarter"] == 0, delta < 0, delta > 0],
        ["churn", "contraction", "expansion"],
        default="flat",
    )
    df["revenue_delta"] = delta
    return df[["account_id", "revenue_end_of_quarter", "revenue_delta", "outcome"]]
