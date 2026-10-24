import numpy as np

from src.data import load_accounts, load_ground_truth, load_raw, tag_transcript


def test_north_america_survives_loading():
    df = load_raw()
    assert df["region"].isna().sum() == 0
    assert (df["region"] == "NA").sum() == 402


def test_accounts_have_no_future_columns():
    df = load_accounts()
    assert "revenue_end_of_quarter" not in df.columns
    assert len(df) == 1000
    assert df["account_id"].is_unique


def test_imputation_leaves_flags_behind():
    df = load_accounts()
    assert df["ai_usage"].isna().sum() == 0
    assert df["ai_usage_missing"].sum() == 30
    assert df["days_since_last_sales_activity_missing"].sum() == 20
    assert df["no_recent_call"].sum() == 52


def test_derived_features_in_range():
    df = load_accounts()
    assert df["seat_utilisation"].between(0, 1).all()
    assert df["seats_exceed_employees"].sum() == 33
    assert np.isnan(df.loc[df["seats_exceed_employees"], "seat_headroom"]).all()
    for col in ["seat_utilisation_pct", "ai_usage_pct"]:
        assert df[col].between(0, 1).all()


def test_keyword_tagger():
    assert tag_transcript("Budget cuts discussed. Our tool is on the list.") == "risk"
    assert tag_transcript("They want to roll out to HR, requesting 85 additional seats.") == "growth"
    assert tag_transcript("Friendly call. No risk signals detected.") == "neutral"
    assert tag_transcript(None) == "none"
    assert tag_transcript("") == "none"


def test_ground_truth_outcomes():
    gt = load_ground_truth()
    counts = gt["outcome"].value_counts()
    assert counts["churn"] == 82
    assert counts["contraction"] == 28
    assert counts["expansion"] == 298
    assert counts["flat"] == 592
