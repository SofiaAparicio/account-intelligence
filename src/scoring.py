"""Scoring layer: which accounts need attention, and why.

Two visible scores and one sort key (docs/decisions.md #6, #7):

- risk_score (0-100)         evidence the account is in trouble
- opportunity_score (0-100)  evidence the account wants to grow
- priority_score             max(risk, opportunity) x urgency x value

Every point comes from a bucket in notebooks/01_eda.ipynb section 6, so each
rule can be read aloud. The transcript label is capped at 20 points so a
single sentence cannot outvote three numeric signals (#5).

Run `python -m src.scoring` to print the ground-truth validation report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import load_accounts, load_ground_truth

# --------------------------------------------------------------------------
# Rules. Each rule: (upper bound inclusive, points, reason template).
# Bounds are read left to right; the first matching bucket wins.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    column: str
    buckets: tuple  # ((upper_bound, points, reason), ...) ascending bounds
    max_points: int
    higher_is_worse: bool = True

    def apply(self, value: float) -> tuple[int, str | None]:
        for upper, points, reason in self.buckets:
            if value <= upper:
                return points, (reason.format(v=value) if reason else None)
        return 0, None


RISK_RULES = [
    Rule("days_since_last_sales_activity", max_points=30, buckets=(
        (30, 0, None),
        (60, 5, None),
        (90, 12, "No sales touch in {v:.0f} days"),
        (120, 22, "No sales touch in {v:.0f} days"),
        (10_000, 30, "No sales touch in {v:.0f} days"),
    )),
    Rule("seat_utilisation", max_points=30, higher_is_worse=False, buckets=(
        (0.30, 30, "Only {v:.0%} of licensed seats in use"),
        (0.45, 20, "Only {v:.0%} of licensed seats in use"),
        (0.60, 10, "{v:.0%} of licensed seats in use"),
        (0.75, 3, None),
        (1.01, 0, None),
    )),
    Rule("nr_support_tickets", max_points=20, buckets=(
        (2, 0, None),
        (4, 6, None),
        (6, 9, "{v:.0f} open support tickets"),
        (8, 14, "{v:.0f} open support tickets"),
        (10_000, 20, "{v:.0f} open support tickets"),
    )),
    Rule("ai_usage", max_points=15, higher_is_worse=False, buckets=(
        (0.10, 15, "AI feature adoption very low ({v:.2f})"),
        (0.20, 11, "AI feature adoption low ({v:.2f})"),
        (0.35, 6, None),
        (0.50, 2, None),
        (1.01, 0, None),
    )),
]

OPPORTUNITY_RULES = [
    Rule("seat_utilisation", max_points=30, buckets=(
        (0.45, 0, None),
        (0.60, 5, None),
        (0.75, 14, "{v:.0%} of seats in use"),
        (0.90, 26, "{v:.0%} of seats in use, near seat cap"),
        (1.01, 30, "{v:.0%} of seats in use, at seat cap"),
    )),
    Rule("ai_usage", max_points=25, buckets=(
        (0.35, 0, None),
        (0.50, 8, None),
        (0.70, 17, "Strong AI feature adoption ({v:.2f})"),
        (1.01, 25, "Very strong AI feature adoption ({v:.2f})"),
    )),
    Rule("days_since_last_sales_activity", max_points=20, higher_is_worse=False, buckets=(
        (14, 20, "Active conversation ({v:.0f} days since last touch)"),
        (30, 15, "Recent conversation ({v:.0f} days since last touch)"),
        (60, 8, None),
        (10_000, 0, None),
    )),
    Rule("nr_support_tickets", max_points=10, higher_is_worse=False, buckets=(
        (0, 10, "No open support tickets"),
        (2, 6, None),
        (4, 2, None),
        (10_000, 0, None),
    )),
]

LABEL_POINTS = {  # capped contribution of the transcript label
    "risk": 20,
    "none": 8,  # no transcript at all doubles churn rate (EDA section 6)
}
GROWTH_LABEL_POINTS = 20

RISK_TYPE_TEXT = {
    "budget": "Last call: budget pressure / cost reduction",
    "competitor": "Last call: competitor in evaluation",
    "champion_left": "Last call: champion left, leadership reviewing vendors",
    "support_escalation": "Last call: unresolved support escalation",
    "roi_doubt": "Last call: doubts about ROI / adoption",
    "consolidation": "Last call: merger or move to in-house tool",
    "unspecified": "Last call contained risk language",
}
EXPANSION_TYPE_TEXT = {
    "seats": "Last call: asked about additional seats",
    "new_team": "Last call: wants to roll out to another team",
    "module": "Last call: interested in an additional module",
    "unspecified": "Last call contained expansion language",
}

RISK_FOCUS = 60
OPP_FOCUS = 50

# Urgency: renewal proximity changes the deadline, not the diagnosis (#7)
URGENCY = ((45, 1.5), (90, 1.3), (135, 1.15), (180, 1.0), (10_000, 0.9))


def _urgency(days: float) -> float:
    for upper, mult in URGENCY:
        if days <= upper:
            return mult
    return 0.9


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def _score_row(row: pd.Series, rules: list[Rule]) -> tuple[int, list[tuple[int, str]]]:
    total, reasons = 0, []
    for rule in rules:
        pts, reason = rule.apply(row[rule.column])
        total += pts
        if reason and pts > 0:
            reasons.append((pts, reason))
    return total, reasons


def score_accounts(df: pd.DataFrame) -> pd.DataFrame:
    """Add risk_score, opportunity_score, priority_score, focus and reasons."""
    out = df.copy()
    risk, opp, risk_reasons, opp_reasons = [], [], [], []

    for _, row in out.iterrows():
        r, rr = _score_row(row, RISK_RULES)
        o, orr = _score_row(row, OPPORTUNITY_RULES)

        intent = row["intent"]
        if intent in LABEL_POINTS:
            pts = LABEL_POINTS[intent]
            r += pts
            text = (
                RISK_TYPE_TEXT.get(row.get("risk_type", "unspecified"), RISK_TYPE_TEXT["unspecified"])
                if intent == "risk" else "No recent call summary on file"
            )
            rr.append((pts, text))
        elif intent == "growth":
            o += GROWTH_LABEL_POINTS
            orr.append((GROWTH_LABEL_POINTS, EXPANSION_TYPE_TEXT.get(
                row.get("expansion_type", "unspecified"), EXPANSION_TYPE_TEXT["unspecified"])))

        risk.append(min(r, 100))
        opp.append(min(o, 100))
        risk_reasons.append([t for _, t in sorted(rr, reverse=True)])
        opp_reasons.append([t for _, t in sorted(orr, reverse=True)])

    out["risk_score"] = risk
    out["opportunity_score"] = opp
    out["risk_reasons"] = risk_reasons
    out["opportunity_reasons"] = opp_reasons

    out["urgency"] = out["days_to_next_renewal"].map(_urgency)
    log_rev = np.log10(out["current_revenue"])
    out["value"] = 0.5 + (log_rev - log_rev.min()) / (log_rev.max() - log_rev.min())  # 0.5 .. 1.5

    base = out[["risk_score", "opportunity_score"]].max(axis=1)
    out["priority_score"] = (base * out["urgency"] * out["value"]).round(1)

    # Focus thresholds chosen on the ground truth so each bucket means something:
    # risk >= 60 -> 188 accounts, 47% actually churn/contract (81% of all churn)
    # opportunity >= 50 -> 233 accounts, 87% actually expand (68% of all expansion)
    out["focus"] = np.select(
        [
            (out["risk_score"] >= RISK_FOCUS) & (out["risk_score"] >= out["opportunity_score"]),
            (out["opportunity_score"] >= OPP_FOCUS) & (out["opportunity_score"] > out["risk_score"]),
            (out["risk_score"] >= 40) | (out["opportunity_score"] >= 40),
        ],
        ["risk", "opportunity", "watch"],
        default="healthy",
    )
    out["reasons"] = np.where(out["focus"] == "opportunity", out["opportunity_reasons"], out["risk_reasons"])
    return out.sort_values("priority_score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Validation against ground truth (never used to build the score)
# --------------------------------------------------------------------------
def validate_against_ground_truth(scored: pd.DataFrame, gt: pd.DataFrame | None = None,
                                  top_share: float = 0.10) -> dict:
    """How much of the real churn / expansion lands in the top slice?"""
    gt = load_ground_truth() if gt is None else gt
    m = scored.merge(gt, on="account_id")
    k = int(round(len(m) * top_share))
    n_bad = (m["outcome"].isin(["churn", "contraction"])).sum()
    n_exp = (m["outcome"] == "expansion").sum()

    def capture(sort_col: str, outcomes: list[str], n_total: int) -> dict:
        top = m.nlargest(k, sort_col)
        hits = top["outcome"].isin(outcomes).sum()
        return {"captured": int(hits), "of": int(n_total), "recall": round(float(hits / n_total), 3),
                "precision": round(float(hits / k), 3)}

    rev_at_risk_top = m.nlargest(k, "risk_score")
    report = {
        "top_k": k,
        "risk_score": capture("risk_score", ["churn", "contraction"], n_bad),
        "opportunity_score": capture("opportunity_score", ["expansion"], n_exp),
        "baseline_revenue_sorted_churn": capture("current_revenue", ["churn", "contraction"], n_bad),
        "baseline_revenue_sorted_expansion": capture("current_revenue", ["expansion"], n_exp),
        "random_recall": round(top_share, 3),
        "revenue_lost_captured_by_top_risk": round(float(
            -rev_at_risk_top.loc[rev_at_risk_top["revenue_delta"] < 0, "revenue_delta"].sum()
            / -m.loc[m["revenue_delta"] < 0, "revenue_delta"].sum()), 3),
    }

    # Lift by decile (10 = highest score): churn/contraction rate per risk decile,
    # expansion rate per opportunity decile
    m["risk_decile"] = pd.qcut(m["risk_score"].rank(method="first"), 10, labels=range(1, 11))
    m["opp_decile"] = pd.qcut(m["opportunity_score"].rank(method="first"), 10, labels=range(1, 11))
    report["churn_rate_by_risk_decile"] = {
        int(k): float(v) for k, v in m.groupby("risk_decile", observed=True)["outcome"]
        .apply(lambda s: s.isin(["churn", "contraction"]).mean()).round(3).items()}
    report["expansion_rate_by_opportunity_decile"] = {
        int(k): float(v) for k, v in m.groupby("opp_decile", observed=True)["outcome"]
        .apply(lambda s: (s == "expansion").mean()).round(3).items()}
    return report


def logistic_sanity_check(scored: pd.DataFrame, gt: pd.DataFrame | None = None) -> pd.Series:
    """Standardised logistic-regression coefficients for churn (numpy, no sklearn).

    Used only to confirm the hand-set signals point the right way and rank in a
    similar order. Not used in the score (docs/decisions.md #8).
    """
    gt = load_ground_truth() if gt is None else gt
    m = scored.merge(gt, on="account_id")
    feats = pd.DataFrame({
        "days_since_last_sales_activity": m["days_since_last_sales_activity"],
        "low_seat_utilisation": -m["seat_utilisation"],
        "nr_support_tickets": m["nr_support_tickets"],
        "low_ai_usage": -m["ai_usage"],
        "risk_language": (m["intent"] == "risk").astype(float),
        "no_recent_call": m["no_recent_call"].astype(float),
        "days_to_next_renewal": m["days_to_next_renewal"],
    })
    X = ((feats - feats.mean()) / feats.std()).to_numpy()
    X = np.c_[np.ones(len(X)), X]
    y = m["outcome"].isin(["churn", "contraction"]).to_numpy(dtype=float)
    w = np.zeros(X.shape[1])
    for _ in range(3000):  # plain gradient descent with light L2
        p = 1 / (1 + np.exp(-X @ w))
        w -= 0.1 * (X.T @ (p - y) / len(y) + 1e-3 * np.r_[0, w[1:]])
    return pd.Series(w[1:], index=feats.columns).round(3).sort_values(ascending=False)


def main() -> None:
    scored = score_accounts(load_accounts())
    report = validate_against_ground_truth(scored)
    print("Top-decile validation (k =", report["top_k"], ")")
    for key in ["risk_score", "opportunity_score", "baseline_revenue_sorted_churn",
                "baseline_revenue_sorted_expansion"]:
        print(f"  {key:36s} {report[key]}")
    print("  random recall                       ", report["random_recall"])
    print("  share of lost revenue in top risk   ", report["revenue_lost_captured_by_top_risk"])
    print("\nChurn/contraction rate by risk decile (10 = highest risk):")
    print("  ", report["churn_rate_by_risk_decile"])
    print("Expansion rate by opportunity decile:")
    print("  ", report["expansion_rate_by_opportunity_decile"])
    print("\nLogistic-regression sanity check (standardised coefficients for churn):")
    print(logistic_sanity_check(scored).to_string())
    print("\nFocus split:", scored["focus"].value_counts().to_dict())


if __name__ == "__main__":
    main()
