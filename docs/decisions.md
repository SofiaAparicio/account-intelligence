# Key decisions

A running log. Each entry: what we chose, what we rejected, why.

## Data

### 1. Read `region` with `keep_default_na=False`
`NA` is North America, not missing. Default pandas parsing silently drops 402
of 1000 accounts (40%). Found in `notebooks/01_eda.ipynb`, section 1.

### 2. `revenue_end_of_quarter` is ground truth, not a feature
It is the future. It lives in a separate table used only to validate that the
score ranks the accounts that actually churned / expanded near the top.

### 3. Missing transcript is a signal, not a gap
Accounts with no call summary churn at 17% vs 8%. We keep it as a
`no_recent_call` flag instead of imputing. Missing `ai_usage` and activity
days are imputed with the segment median plus a missing flag.

## Scoring

### 4. Labeler → ranker → briefer
The intelligence layer has three parts: an LLM **labeler** turns each call
transcript into typed labels (intent, risk type, expansion type); a
deterministic **ranker** scores accounts from numeric signals plus those
labels; an LLM **briefer** writes the meeting prep. Rejected: one LLM doing
all three. Labeling and ranking need to be cheap, reproducible and
explainable; writing needs to be generative. Different jobs, different tools.

*Future:* the labeler and (optionally) the ranker are the slots a
non-generative decision model such as TypeSafe's Jev would fill: typed
answers with probabilities, ~0.4 s and a fraction of a cent per decision.
It cannot write the brief.

### 5. Transcript labels are a real, weight-capped feature
Risk language → 65% churn or contraction, growth language → 62% expansion:
as strong as the best numeric signals. The label weight is capped so one
sentence cannot outvote three numeric signals, and it always appears as a
reason code so the AE sees why it moved the rank.

### 5b. The transcripts are a near-perfect oracle in this dataset, and we say so
After labelling, every "risk" transcript churned or contracted, every
"growth" transcript expanded, every "neutral" one stayed flat. The synthetic
generator evidently derived the outcome from the call template. Uncapped, the
label alone would "solve" the dataset, which real call notes never do. The
20-point cap is therefore not just a safety measure: it is what keeps the
ranker honest about the numeric signals an AE would rely on in production.
Labels moved top-decile risk recall from 64% to 71% (precision 70% to 78%).

### 6. Two visible scores, one sort key
Risk (0–100) and opportunity (0–100) are shown separately because an AE acts
differently on "save" vs "grow". A single `priority` =
max(risk, opportunity) × urgency × log(revenue) drives the default order.
Rejected: one blended number (hides which action is needed).

### 7. Renewal proximity is urgency, not risk
Churn is 8–13% in every renewal window; an account untouched for 90+ days
churns at ~25% whether or not renewal is close. Days to renewal therefore
multiplies priority (how soon to act) but adds nothing to the risk score
(whether something is wrong).

### 8. Hand-set weights from the EDA, logistic regression as sanity check only
Weights and thresholds come from the bucketed churn / expansion rates in the
EDA, so every rule can be read aloud. A small logistic regression on the
ground truth is run only to confirm the signals rank in the same order.
Rejected: fitting the model *as* the score (trains on the outcome we are
predicting; not explainable to an AE).

*Result (`python -m src.scoring`):* the top risk decile contains 64% of all
churn/contraction at 70% precision (revenue-sorted baseline: 8%); the top
opportunity decile is 98% actual expanders. The regression ranks the same
signals in the same direction and gives renewal proximity a coefficient near
zero, confirming #7. It would weight transcript risk language *higher* than
our 20-point cap; we keep the cap on purpose (#5).

### 9. Industry stays out of the score
Healthcare churns at 17%, Technology at 4%, but this is the signal most
likely to be a synthetic-data artefact and it penalises a sector rather than
a behaviour. It is shown in the peer-comparison context instead.

## LLM

### 10. No API keys; the model in the loop is the Claude session that built this
Labels for the 447 distinct transcripts and briefs for the top accounts are
generated once and committed, each with the exact prompt and context card
that produced it. The app runs with no key. A thin optional adapter
(Ollama, or any provider the user configures) enables live regeneration;
otherwise the UI shows the assembled prompt with a copy button.

### 11. Brief is a fixed JSON schema with an `unknowns` section
`situation`, `risks`, `opportunities`, `talking_points` (3), `next_action`,
`unknowns`. Prompt rules: only numbers from the context card, no invented
names or dates, cite the signal behind each claim. `unknowns` is what the
data cannot tell us; it doubles as the AE's question list for the meeting.

## UI

### 12. Two screens, no dashboard
*Today's focus* (filters, KPI strip, ranked table with score chips and
reason codes) and *Meeting brief* (facts, score breakdown, brief, prompt).
The brief asks for prioritised accounts and the insight, graded on AE
usability. Portfolio charts answer a manager's question, not the AE's.

### 13. Filters stand in for the AE's book of business
Segment / region / industry filters let the demo say "I'm the EMEA
Mid-Market AE". Rejected: inventing an `owner` column (fabricated data).

### 14. Streamlit
One language, runnable from the README with one command.
