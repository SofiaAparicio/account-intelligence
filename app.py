"""Account Intelligence - Streamlit entry point.

Two views, one per AE question:

1. Today's focus   - ranked accounts with score chips, reason codes, filters
2. Meeting brief   - pick an account: facts, score breakdown, AI-generated brief

Run:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="Account Intelligence", layout="wide")
st.title("Account Intelligence")
st.info("Scaffold only. Data, scoring and insight layers are implemented in src/.")
