# Account Intelligence

A small app that helps an Account Executive answer two questions at the start
of the day: **which accounts should I focus on**, and **what should I know
before my next meeting**.

## What it does

- Ranks ~1,000 accounts by a transparent **risk score** and **opportunity
  score** (0–100 each), every point traceable to a readable rule, with the
  reasons shown next to each account.
- Shows two lists, **To save** and **Ready to grow**, ordered by priority
  (evidence × renewal urgency × revenue), with filters that stand in for the
  AE's book of business.
- Opens a **meeting brief** for any account: situation, risks, opportunities,
  talking points, next action, and what the data can't tell you — grounded
  only in that account's data, with the exact prompt shown alongside.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --reload
# open http://localhost:8000
```

No API key is needed. The AI-generated labels and briefs are committed to the
repo. To regenerate a brief live, run [Ollama](https://ollama.com) locally and
set `OLLAMA_MODEL` (e.g. `llama3.1`) in `.env`.

```bash
pytest            # tests for data, scoring, labeler and briefs
python -m src.scoring     # ground-truth validation report
```

## Architecture

![Architecture](docs/architecture.png)

Purple boxes are the only places a language model is involved; both run
offline and their outputs are committed with the prompts that produced them.
See [docs/ai_usage.md](docs/ai_usage.md) for a short account of how AI was
used, and [docs/decisions.md](docs/decisions.md) for every design decision
and what was rejected.

## Repository layout

```
.
├── api.py                 FastAPI: JSON routes + serves the web page
├── web/index.html         the UI (one page, vanilla JS, no build step)
├── src/
│   ├── data.py            load, clean, derive features
│   ├── scoring.py         risk / opportunity / priority + reason codes
│   ├── labeler.py         expands the LLM transcript labels to every account
│   ├── prompts.py         labeler prompt, brief prompt, context card
│   ├── insights.py        reads committed briefs; optional live path
│   └── api_core.py        payload logic behind the routes (unit-testable)
├── data/
│   ├── raw/               account_data.csv (challenge input)
│   ├── labels/            LLM labels per call template, and per account
│   └── briefs/            22 committed meeting briefs, each with its prompt
├── notebooks/01_eda.ipynb exploratory analysis that set the scoring weights
├── docs/
│   ├── decisions.md       key decisions and why
│   ├── ai_usage.md        how AI was used (overview)
│   └── architecture.png   the diagram above (python docs/make_architecture.py)
└── tests/
```

## How the ranking works, in one paragraph

Four numeric signals (seat utilisation, days since last sales touch, open
support tickets, AI feature adoption) and one call-transcript label each add
points from a bucket table set in the EDA. Risk and opportunity are scored
separately because an AE acts differently on "save" and "grow". Days to
renewal is *urgency*, not risk (churn is flat across renewal windows in this
data), so it multiplies the priority rather than the score. On the held-out
`revenue_end_of_quarter`, the top risk decile captures 71% of all churn and
contraction at 78% precision; the top opportunity decile is 100% actual
expanders. A revenue-sorted list, the obvious baseline, captures 8%.

## What I'd do with more time

- Replace the labeler with a typed decision model (e.g. Jev) and compare a
  learned score against the hand-set ranker on the ground truth.
- Let the AE mark an account as "handled" so it drops off today's list, and
  keep a small history of score changes per account.
- Pull real call transcripts instead of summaries; the current labels are as
  good as they are partly because the summaries are templated.
- A manager view (portfolio charts) was deliberately left out; it answers a
  different question than the AE's.
