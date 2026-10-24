"""Data layer: load, clean and enrich the account dataset.

Responsibilities
----------------
- load_raw():        read data/raw/account_data.csv *correctly* (see NA trap below)
- clean():           handle nulls, flag anomalies, enforce types
- add_features():    derived columns used by scoring and by the LLM context card
- load_accounts():   the one function the app calls; returns the enriched frame
- load_ground_truth(): revenue_end_of_quarter kept separate, for validation ONLY

Known data traps (from notebooks/01_eda.ipynb)
----------------------------------------------
- region "NA" means North America. pandas' default NA parsing turns it into NaN
  and silently drops 40% of accounts. Always read with keep_default_na=False.
- revenue_end_of_quarter is a *future* outcome. Never use it as a feature.
- ai_usage (30), days_since_last_sales_activity (20) and
  call_transcript_summary (52) have real missing values.
- 33 accounts have more licensed seats than employees.
"""

# TODO(step 2): implement after EDA is reviewed
