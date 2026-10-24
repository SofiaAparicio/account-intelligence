import pandas as pd

from src.data import load_accounts
from src.scoring import score_accounts, validate_against_ground_truth


def _scored():
    return score_accounts(load_accounts())


def test_scores_in_range_and_sorted():
    s = _scored()
    assert s["risk_score"].between(0, 100).all()
    assert s["opportunity_score"].between(0, 100).all()
    assert s["priority_score"].is_monotonic_decreasing
    assert set(s["focus"]) <= {"risk", "opportunity", "watch", "healthy"}


def test_every_scored_account_has_reasons_when_flagged():
    s = _scored()
    flagged = s[s["focus"].isin(["risk", "opportunity"])]
    assert flagged["reasons"].map(len).min() >= 1
    # focus buckets should be selective enough to act on
    assert 150 <= (s["focus"] == "risk").sum() <= 250
    assert 180 <= (s["focus"] == "opportunity").sum() <= 280


def test_textbook_risk_account_scores_high():
    row = pd.DataFrame([{
        "account_id": "X", "days_since_last_sales_activity": 150, "seat_utilisation": 0.2,
        "nr_support_tickets": 10, "ai_usage": 0.05, "intent": "risk", "risk_type": "competitor",
        "expansion_type": "none", "days_to_next_renewal": 40, "current_revenue": 50_000.0,
    }])
    # value normalisation needs a range; add a healthy twin
    twin = row.copy()
    twin.update({"account_id": ["Y"], "days_since_last_sales_activity": [5], "seat_utilisation": [0.95],
                 "nr_support_tickets": [0], "ai_usage": [0.9], "intent": ["growth"], "risk_type": ["none"],
                 "expansion_type": ["seats"], "days_to_next_renewal": [250], "current_revenue": [500_000.0]})
    s = score_accounts(pd.concat([row, twin]))
    bad, good = s.set_index("account_id").loc["X"], s.set_index("account_id").loc["Y"]
    assert bad["risk_score"] == 100 and bad["opportunity_score"] == 0 and bad["focus"] == "risk"
    assert good["opportunity_score"] == 100 and good["risk_score"] == 0 and good["focus"] == "opportunity"
    assert "competitor" in " ".join(bad["reasons"]).lower()


def test_transcript_label_is_capped():
    base = {"account_id": "A", "days_since_last_sales_activity": 5, "seat_utilisation": 0.9,
            "nr_support_tickets": 0, "ai_usage": 0.9, "intent": "risk", "risk_type": "budget",
            "expansion_type": "none", "days_to_next_renewal": 100, "current_revenue": 10_000.0}
    other = dict(base, account_id="B", intent="neutral", current_revenue=20_000.0)
    s = score_accounts(pd.DataFrame([base, other])).set_index("account_id")
    assert s.loc["A", "risk_score"] == 20  # label alone cannot push a healthy account into focus
    assert s.loc["A", "focus"] == "opportunity"


def test_beats_random_and_revenue_baseline_on_ground_truth():
    r = validate_against_ground_truth(_scored())
    assert r["risk_score"]["recall"] > 3 * r["random_recall"]
    assert r["opportunity_score"]["recall"] > 2 * r["random_recall"]
    assert r["risk_score"]["recall"] > r["baseline_revenue_sorted_churn"]["recall"]
    assert r["opportunity_score"]["recall"] > r["baseline_revenue_sorted_expansion"]["recall"]
