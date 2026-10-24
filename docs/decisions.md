# Key decisions

A running log. Each entry: what we chose, what we rejected, why.

## 1. Read `region` with `keep_default_na=False`
`NA` is North America, not missing. Default pandas parsing drops 402 of 1000
accounts. Found in EDA (notebooks/01_eda.ipynb).

## 2. `revenue_end_of_quarter` is ground truth, not a feature
It is the future. We use it only to validate that the heuristic score ranks
the accounts that actually churned / expanded near the top.

## 3. Rule-based scoring, not a trained model
Transparent, explainable to an AE, fits the 3-4h budget, and the EDA shows the
signals separate cleanly without a model.

## 4. Two scores (risk, opportunity) rather than one blended score
An AE takes different actions on "save" vs "grow". A single number hides which
one is driving the rank.

## 5. Streamlit
One language, runnable from the README with one command.

## 6. LLM output cached to disk
The live demo must not depend on an API call succeeding.

<!-- append as we go -->
