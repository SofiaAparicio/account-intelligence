import pandas as pd

from src.data import load_ground_truth, load_raw
from src.labeler import label_transcripts, normalise


def test_normalise_collapses_variable_slots():
    a = normalise("Exciting call with CTO. They want to roll out to HR — requesting 85 additional seats.")
    b = normalise("Exciting call with CFO. They want to roll out to operations — requesting 5 additional seats.")
    assert a == b
    assert "<ROLE>" in a and "<TEAM>" in a and "<N>" in a


def test_every_transcript_gets_a_template_label():
    labels = label_transcripts(load_raw())
    assert len(labels) == 1000
    assert (labels["label_source"] == "keyword").sum() == 0, "a template is missing from template_labels.json"
    assert (labels["label_source"] == "no_transcript").sum() == 52


def test_labels_agree_with_ground_truth_direction():
    m = label_transcripts(load_raw()).merge(load_ground_truth(), on="account_id")
    risk = m[m["intent"] == "risk"]["outcome"]
    growth = m[m["intent"] == "growth"]["outcome"]
    assert risk.isin(["churn", "contraction"]).mean() > 0.5
    assert (risk == "expansion").sum() == 0
    assert (growth == "expansion").mean() > 0.5
    assert growth.isin(["churn", "contraction"]).sum() == 0
