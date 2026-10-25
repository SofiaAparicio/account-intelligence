"""Account Intelligence - Streamlit entry point.

Two views, one per AE question (docs/decisions.md #12):

1. Today's focus   - ranked accounts with score chips, reason codes, filters
2. Meeting brief   - pick an account: facts, score breakdown, AI-generated brief

Run:  streamlit run app.py
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from src.data import load_accounts
from src.insights import NotConfigured, build_prompt, generate_brief, get_brief, segment_medians
from src.scoring import OPP_FOCUS, RISK_FOCUS, score_accounts

st.set_page_config(page_title="Account Intelligence", page_icon="◎", layout="wide")

FOCUS_LABEL = {"risk": "🔴 Risk", "opportunity": "🟢 Opportunity", "watch": "🟠 Watch", "healthy": "⚪ Healthy"}


@st.cache_data(show_spinner="Scoring accounts…")
def load_scored() -> pd.DataFrame:
    return score_accounts(load_accounts())


def money(x: float) -> str:
    return f"${x/1e6:.1f}M" if x >= 1e6 else f"${x/1e3:.0f}K"


scored = load_scored()

# --------------------------------------------------------------------------
# Sidebar: the AE's book of business (#13)
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("Account Intelligence")
    st.caption("Which accounts should I focus on, and what should I know before my next meeting?")
    st.subheader("My book")
    segs = st.multiselect("Segment", sorted(scored["segment"].unique()), placeholder="All segments")
    regs = st.multiselect("Region", sorted(scored["region"].unique()), placeholder="All regions")
    inds = st.multiselect("Industry", sorted(scored["industry"].unique()), placeholder="All industries")
    focus_sel = st.multiselect("Focus", list(FOCUS_LABEL), default=["risk", "opportunity"],
                               format_func=FOCUS_LABEL.get)
    st.divider()
    st.caption(
        f"Risk ≥ {RISK_FOCUS} and Opportunity ≥ {OPP_FOCUS} define the focus buckets. "
        "Scores are rule-based and every point has a reason; see README."
    )

book = scored.copy()
if segs:
    book = book[book["segment"].isin(segs)]
if regs:
    book = book[book["region"].isin(regs)]
if inds:
    book = book[book["industry"].isin(inds)]
view = book[book["focus"].isin(focus_sel)] if focus_sel else book

tab_focus, tab_brief = st.tabs(["Today's focus", "Meeting brief"])

# --------------------------------------------------------------------------
# View 1: Today's focus
# --------------------------------------------------------------------------
with tab_focus:
    at_risk = book[book["focus"] == "risk"]
    growing = book[book["focus"] == "opportunity"]
    soon = book[(book["focus"].isin(["risk", "opportunity"])) & (book["days_to_next_renewal"] <= 45)]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accounts at risk", f"{len(at_risk)}", f"of {len(book)} in my book", delta_color="off")
    k2.metric("Revenue at risk", money(at_risk["current_revenue"].sum()))
    k3.metric("Expansion candidates", f"{len(growing)}", money(growing["current_revenue"].sum()) + " base",
              delta_color="off")
    k4.metric("Renewals ≤ 45 days needing action", f"{len(soon)}")

    st.caption(
        f"Showing {len(view)} accounts sorted by priority "
        "(= max(risk, opportunity) × renewal urgency × log revenue). Click a row to open its brief."
    )

    show_all = st.toggle("Show all matching accounts", value=False)
    table = view if show_all else view.head(25)
    table = table.assign(
        rank=range(1, len(table) + 1),
        focus_label=table["focus"].map(FOCUS_LABEL),
        why=table["reasons"].map(lambda r: " · ".join(r[:2]) if r else "—"),
    )

    event = st.dataframe(
        table[["rank", "account_name", "segment", "region", "focus_label", "risk_score",
               "opportunity_score", "days_to_next_renewal", "current_revenue", "why", "account_id"]],
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "rank": st.column_config.NumberColumn("#", width="small"),
            "account_name": st.column_config.TextColumn("Account", width="medium"),
            "segment": "Segment",
            "region": "Region",
            "focus_label": st.column_config.TextColumn("Focus", width="small"),
            "risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%d"),
            "opportunity_score": st.column_config.ProgressColumn("Opportunity", min_value=0, max_value=100, format="%d"),
            "days_to_next_renewal": st.column_config.NumberColumn("Renewal in", format="%d d"),
            "current_revenue": st.column_config.NumberColumn("ARR", format="$%d"),
            "why": st.column_config.TextColumn("Why", width="large"),
            "account_id": None,
        },
        height=min(38 * (len(table) + 1), 900),
    )
    rows = event.selection.rows if event and event.selection else []
    if rows:
        st.session_state["selected_account"] = table.iloc[rows[0]]["account_id"]
        st.info(f"Selected **{table.iloc[rows[0]]['account_name']}** — open the *Meeting brief* tab.")

# --------------------------------------------------------------------------
# View 2: Meeting brief
# --------------------------------------------------------------------------
with tab_brief:
    options = view["account_id"].tolist() or scored["account_id"].tolist()
    default = st.session_state.get("selected_account", options[0])
    if default not in options:
        options = [default] + options
    names = scored.set_index("account_id")["account_name"]
    account_id = st.selectbox("Account", options, index=options.index(default),
                              format_func=lambda a: f"{names[a]}  ({a})")
    st.session_state["selected_account"] = account_id

    acc = scored.set_index("account_id").loc[account_id].to_dict() | {"account_id": account_id}

    st.header(acc["account_name"])
    st.caption(acc["account_description"])
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Focus", FOCUS_LABEL[acc["focus"]])
    c2.metric("Risk", f"{acc['risk_score']}/100")
    c3.metric("Opportunity", f"{acc['opportunity_score']}/100")
    c4.metric("Renewal in", f"{acc['days_to_next_renewal']} days")
    c5.metric("ARR", money(acc["current_revenue"]))

    left, right = st.columns([1, 1.4])

    with left:
        st.subheader("Why it's ranked here")
        if acc["risk_reasons"]:
            st.markdown("**Risk signals**")
            for r in acc["risk_reasons"]:
                st.markdown(f"- {r}")
        if acc["opportunity_reasons"]:
            st.markdown("**Growth signals**")
            for r in acc["opportunity_reasons"]:
                st.markdown(f"- {r}")
        if not acc["risk_reasons"] and not acc["opportunity_reasons"]:
            st.markdown("No strong signals either way.")

        st.subheader("Facts")
        facts = {
            "Industry / Segment / Region": f"{acc['industry']} / {acc['segment']} / {acc['region']}",
            "Employees": f"{acc['nr_employees']:,}",
            "Seats (licensed / active)": f"{acc['nr_licensed_seats']:,} / {acc['nr_active_users']:,}  "
                                         f"({acc['seat_utilisation']:.0%}, P{int(acc['seat_utilisation_pct']*100)} in segment)",
            "Days since last sales activity": f"{acc['days_since_last_sales_activity']}"
                                              + ("  (imputed)" if acc.get("days_since_last_sales_activity_missing") else ""),
            "Open support tickets": f"{acc['nr_support_tickets']}",
            "AI feature adoption": f"{acc['ai_usage']:.2f}" + ("  (imputed)" if acc.get("ai_usage_missing") else ""),
            "Last call": acc["call_transcript_summary"] if isinstance(acc.get("call_transcript_summary"), str)
                         else "No call summary on file",
            "Call label": f"{acc['intent']} · {acc.get('risk_type', 'none')} · {acc.get('expansion_type', 'none')}",
        }
        st.table(pd.Series(facts, name="").to_frame())

    with right:
        st.subheader("Meeting brief")
        cached = get_brief(account_id)
        peers = segment_medians(scored, acc["segment"])
        prompt, card = build_prompt(acc, peers)

        live_ok = bool(os.getenv("OLLAMA_MODEL"))
        if st.button("Generate with local model" if live_ok else "Generate (no local model configured)",
                      disabled=not live_ok, help="Set OLLAMA_MODEL with Ollama running to enable. Optional."):
            try:
                with st.spinner("Asking the model…"):
                    st.session_state[f"live_{account_id}"] = generate_brief(acc, peers)
            except (NotConfigured, Exception) as e:  # noqa: BLE001 - surface any provider error to the AE
                st.error(str(e))

        brief = st.session_state.get(f"live_{account_id}") or (cached["brief"] if cached else None)
        source = ("live local model" if st.session_state.get(f"live_{account_id}")
                  else (cached["generator"].split(";")[0] if cached else None))

        if brief:
            st.caption(f"Generated by {source}" + (f" on {cached['generated_on']}" if cached and not st.session_state.get(f'live_{account_id}') else ""))
            st.markdown(brief["situation"])
            b1, b2 = st.columns(2)
            with b1:
                st.markdown("**Risks**")
                for r in brief["risks"] or ["None identified."]:
                    st.markdown(f"- {r}")
            with b2:
                st.markdown("**Opportunities**")
                for r in brief["opportunities"] or ["None identified."]:
                    st.markdown(f"- {r}")
            st.markdown("**Talking points**")
            for i, t in enumerate(brief["talking_points"], 1):
                st.markdown(f"{i}. {t}")
            st.success(f"**Next action:** {brief['next_action']}")
            st.markdown("**What the data can't tell us**")
            for u in brief["unknowns"]:
                st.markdown(f"- {u}")
        else:
            st.warning(
                "No pre-generated brief for this account and no local model configured. "
                "The prompt below is exactly what the model would receive; copy it into any LLM."
            )

        with st.expander("Prompt and context card sent to the model", expanded=not brief):
            st.code(prompt, language="markdown")
