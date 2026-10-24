"""Labeler: typed labels for each call transcript.

The LLM judgment lives in data/labels/template_labels.json: each semantic
template of the (highly templated) call summaries was read once and labelled
against the schema in src/prompts.py::LABELER_PROMPT. This module only does
the mechanical part: normalise a transcript to its template, look the label
up, and fall back to the keyword tagger when nothing matches.

Why templates and not 948 individual calls? Because the summaries collapse to
~45 distinct meanings; labelling each meaning once is cheaper, auditable
(you can read every judgment in the JSON) and exactly what a decision model
such as Jev would be asked to do per call in production.

Run `python -m src.labeler` to (re)generate data/labels/transcript_labels.csv.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .data import LABELS_PATH, ROOT, load_raw, tag_transcript

TEMPLATE_PATH = ROOT / "data" / "labels" / "template_labels.json"

PRODUCTS = ["dashboard module", "analytics suite", "automation engine", "Slack integration",
            "reporting tool", "API access"]
COMPETITORS = ["Salesforce", "HubSpot", "Gong", "Clari", "Outreach", "ZoomInfo", "6sense"]
TEAMS = ["customer success", "engineering", "marketing", "operations", "finance", "support",
         "product", "legal", "sales", "HR", "IT"]
ROLE_RE = re.compile(
    r"(VP of [A-Z][a-zA-Z ]*?|Head of [A-Z][a-zA-Z ]*?|Director of [A-Z][a-zA-Z ]*?|CTO|CFO|CMO|COO|CEO|CIO|CRO)"
    r"(?=[\s.,;'])"
)


def normalise(text: str) -> str:
    s = text
    for p in sorted(PRODUCTS, key=len, reverse=True):
        s = re.sub(re.escape(p), "<PRODUCT>", s, flags=re.I)
    for c in COMPETITORS:
        s = re.sub(r"\b" + re.escape(c) + r"\b", "<COMPETITOR>", s)
    s = ROLE_RE.sub("<ROLE>", s)
    s = re.sub(r"\d+%", "<PCT>", s)
    s = re.sub(r"\b\d+\b", "<N>", s)
    for t in sorted(TEAMS, key=len, reverse=True):
        s = re.sub(r"\b" + re.escape(t) + r"\b", "<TEAM>", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()


def load_templates(path: Path = TEMPLATE_PATH) -> dict[str, dict]:
    data = json.loads(Path(path).read_text())
    return {normalise(t["template"]): t for t in data["templates"]}


def label_transcripts(df: pd.DataFrame) -> pd.DataFrame:
    templates = load_templates()
    rows = []
    for acc, text in zip(df["account_id"], df["call_transcript_summary"]):
        if not isinstance(text, str) or not text.strip():
            rows.append((acc, "none", "none", "none", 1.0, "no_transcript", ""))
            continue
        hit = templates.get(normalise(text))
        if hit:
            rows.append((acc, hit["intent"], hit["risk_type"], hit["expansion_type"],
                         hit["confidence"], "llm_template", hit["rationale"]))
        else:
            tag = tag_transcript(text)
            rows.append((acc, tag, "unspecified" if tag == "risk" else "none",
                         "unspecified" if tag == "growth" else "none", 0.5, "keyword", ""))
    return pd.DataFrame(rows, columns=[
        "account_id", "intent", "risk_type", "expansion_type", "confidence", "label_source", "rationale"])


def main() -> None:
    df = load_raw()
    labels = label_transcripts(df)
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(LABELS_PATH, index=False)
    print(f"wrote {LABELS_PATH.relative_to(ROOT)} ({len(labels)} rows)")
    print("\nlabel_source:", labels["label_source"].value_counts().to_dict())
    print("intent:", labels["intent"].value_counts().to_dict())
    print("risk_type:", labels.loc[labels["intent"] == "risk", "risk_type"].value_counts().to_dict())
    print("expansion_type:", labels.loc[labels["intent"] == "growth", "expansion_type"].value_counts().to_dict())
    unmatched = df.loc[labels["label_source"] == "keyword", "call_transcript_summary"]
    if len(unmatched):
        print("\nUNMATCHED (fell back to keywords):")
        for s in unmatched.drop_duplicates().head(20):
            print("  -", s)


if __name__ == "__main__":
    main()
