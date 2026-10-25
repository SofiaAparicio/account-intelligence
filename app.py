"""Account Intelligence - Streamlit entry point.

Two views, one per AE question (docs/decisions.md #12):

1. Today's focus   - two short lists: accounts to save, accounts to grow
2. Meeting brief   - one account: facts, why it is ranked, AI-generated brief

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

VIEWS = ["Today's focus", "Meeting brief"]
TOP_N = 10


@st.cache_data(show_spinner="Scoring accounts…")
def load_scored() -> pd.DataFrame:
    return score_accounts(load_accounts())


def money(x: float) -> str:
    return f"${x/1e6:.1f}M" if x >= 1e6 else f"${x/1e3:.0f}K"


def one_liner(row: pd.Series, kind: str) -> str:
    """Plain-English summary of why this account is on the list."""
    reasons = row["risk_reasons"] if kind == "risk" else row["opportunity_reasons"]
    bits = [r[0].lower() + r[1:] for r in reasons[:3]]
    text = "; ".join(bits).replace("last call: ", "last call mentioned ")
    text = text[0].upper() + text[1:] if text else "No strong signals."
    return f"{text}. Renews in {int(row['days_to_next_renewal'])} days."


def open_brief(account_id: str) -> None:
    st.session_state["selected_account"] = account_id
    st.session_state["view"] = "Meeting brief"


scored = load_scored()
if "view" not in st.session_state:
    st.session_state["view"] = VIEWS[0]

# --------------------------------------------------------------------------
# Sidebar: navigation + the AE's book of business (#13)
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("Account Intelligence")
    view_choice = st.radio("View", VIEWS, key="view", label_visibility="collapsed")
    st.divider()
    st.subheader("My book")
    segs = st.multiselect("Segment", sorted(scored["segment"].unique()), placeholder="All segments")
    regs = st.multiselect("Region", sorted(scored["region"].unique()), placeholder="All regions")
    inds = st.multiselect("Industry", sorted(scored["industry"].unique()), placeholder="All industries")

book = scored.copy()
if segs:
    book = book[book["segment"].isin(segs)]
if regs:
    book = book[book["region"].isin(regs)]
if inds:
    book = book[book["industry"].isin(inds)]


def render_list(df: pd.DataFrame, kind: str) -> None:
    score_col = "risk_score" if kind == "risk" else "opportunity_score"
    for _, row in df.iterrows():
        c_name, c_why, c_score, c_btn = st.columns([2.2, 5, 1, 1.2], vertical_alignment="center")
        c_name.markdown(f"**{row['account_name']}**  \n{row['segment']} · {row['region']} · {money(row['current_revenue'])}")
        c_why.markdown(one_liner(row, kind))
        c_score.markdown(f"<div style='text-align:center;font-size:1.6rem;font-weight:600'>{int(row[score_col])}</div>",
                         unsafe_allow_html=True)
        c_btn.button("Open brief", key=f"{kind}-{row['account_id']}", on_click=open_brief,
                     args=(row["account_id"],), use_container_width=True)
        st.divider()


# --------------------------------------------------------------------------
# View 1: Today's focus
# --------------------------------------------------------------------------
if view_choice == "Today's focus":
    st.title("Today's focus")
    st.caption(f"{len(book):,} accounts in my book. Two lists: who needs saving, who is ready to grow. "
               "Each list is sorted by its score; renewal date and revenue break ties.")

    # Ordered by the score the AE sees; priority (renewal urgency x revenue) breaks ties
    save = book[book["focus"] == "risk"].sort_values(["risk_score", "priority_score"], ascending=False)
    grow = book[book["focus"] == "opportunity"].sort_values(["opportunity_score", "priority_score"], ascending=False)

    k1, k2, k3 = st.columns(3)
    k1.metric("Accounts to save", len(save), help=f"Risk score ≥ {RISK_FOCUS}")
    k2.metric("Revenue on those accounts", money(save["current_revenue"].sum()))
    k3.metric("Accounts ready to grow", len(grow), help=f"Opportunity score ≥ {OPP_FOCUS}")

    st.header("🔴 Accounts to save")
    st.caption(f"Top {min(TOP_N, len(save))} of {len(save)}. The number is the **risk score** (0–100): "
               "how much evidence there is that the account is in trouble. Each point comes from a rule you can read in the brief.")
    render_list(save.head(TOP_N), "risk")

    st.header("🟢 Accounts ready to grow")
    st.caption(f"Top {min(TOP_N, len(grow))} of {len(grow)}. The number is the **opportunity score** (0–100): "
               "how much evidence there is that the account wants to buy more.")
    render_list(grow.head(TOP_N), "opportunity")

    with st.expander("Everything else in my book"):
        rest = book[~book["focus"].isin(["risk", "opportunity"])]
        st.caption(f"{len(rest)} accounts with no strong signal either way (watch / healthy).")
        st.dataframe(
            rest[["account_name", "segment", "region", "focus", "risk_score", "opportunity_score",
                  "days_to_next_renewal", "current_revenue"]]
            .rename(columns={"account_name": "Account", "segment": "Segment", "region": "Region", "focus": "Status",
                             "risk_score": "Risk", "opportunity_score": "Opportunity",
                             "days_to_next_renewal": "Renews in (days)", "current_revenue": "ARR ($)"}),
            hide_index=True, use_container_width=True, height=400,
        )

# --------------------------------------------------------------------------
# View 2: Meeting brief
# --------------------------------------------------------------------------
else:
    options = book["account_id"].tolist() or scored["account_id"].tolist()
    default = st.session_state.get("selected_account", options[0])
    if default not in options:
        options = [default] + options
    names = scored.set_index("account_id")["account_name"]
    account_id = st.selectbox("Account", options, index=options.index(default),
                              format_func=lambda a: f"{names[a]}  ({a})")
    st.session_state["selected_account"] = account_id

    acc = scored.set_index("account_id").loc[account_id].to_dict() | {"account_id": account_id}
    kind = {"risk": "Save", "opportunity": "Grow", "watch": "Watch", "healthy": "Healthy"}[acc["focus"]]

    st.title(acc["account_name"])
    st.caption(f"{acc['account_description']}  \n{acc['industry']} · {acc['segment']} · {acc['region']} · "
               f"{acc['nr_employees']:,} employees")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("What to do", kind)
    c2.metric("Risk score", int(acc["risk_score"]))
    c3.metric("Opportunity score", int(acc["opportunity_score"]))
    c4.metric("Renews in", f"{int(acc['days_to_next_renewal'])} days")
    c5.metric("Annual revenue", money(acc["current_revenue"]))

    left, right = st.columns([1, 1.5], gap="large")

    with left:
        st.subheader("Why it's on the list")
        if acc["risk_reasons"]:
            st.markdown("**Trouble signals**")
            for r in acc["risk_reasons"]:
                st.markdown(f"- {r}")
        if acc["opportunity_reasons"]:
            st.markdown("**Growth signals**")
            for r in acc["opportunity_reasons"]:
                st.markdown(f"- {r}")
        if not acc["risk_reasons"] and not acc["opportunity_reasons"]:
            st.markdown("No strong signals either way.")

        st.subheader("The facts")
        facts = {
            "Seats licensed / active": f"{acc['nr_licensed_seats']:,} / {acc['nr_active_users']:,} "
                                       f"({acc['seat_utilisation']:.0%} used)",
            "Days since last sales activity": f"{int(acc['days_since_last_sales_activity'])}"
                                              + (" (estimated)" if acc.get("days_since_last_sales_activity_missing") else ""),
            "Open support tickets": f"{int(acc['nr_support_tickets'])}",
            "AI feature adoption (0–1)": f"{acc['ai_usage']:.2f}" + (" (estimated)" if acc.get("ai_usage_missing") else ""),
            "Last call": acc["call_transcript_summary"] if isinstance(acc.get("call_transcript_summary"), str)
                         else "No call summary on file",
        }
        for k, v in facts.items():
            st.markdown(f"**{k}**  \n{v}")

    with right:
        st.subheader("Meeting brief")
        cached = get_brief(account_id)
        peers = segment_medians(scored, acc["segment"])
        prompt, card = build_prompt(acc, peers)

        live_ok = bool(os.getenv("OLLAMA_MODEL"))
        live_key = f"live_{account_id}"
        if st.button("Regenerate with local model" if live_ok else "Regenerate (needs a local model)",
                      disabled=not live_ok, help="Optional. Set OLLAMA_MODEL with Ollama running to enable."):
            try:
                with st.spinner("Asking the model…"):
                    st.session_state[live_key] = generate_brief(acc, peers)
            except (NotConfigured, Exception) as e:  # noqa: BLE001 - surface any provider error to the AE
                st.error(str(e))

        brief = st.session_state.get(live_key) or (cached["brief"] if cached else None)

        if brief:
            if st.session_state.get(live_key):
                st.caption("Generated just now by your local model.")
            else:
                st.caption(f"Pre-generated on {cached['generated_on']} by {cached['generator'].split(' (')[0]}, "
                           "from the context card below.")
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
            st.markdown("**What the data can't tell us — ask in the meeting**")
            for u in brief["unknowns"]:
                st.markdown(f"- {u}")
        else:
            st.warning("No pre-generated brief for this account and no local model configured. "
                       "The prompt below is exactly what the model would receive; paste it into any LLM.")

        with st.expander("Prompt and context card sent to the model", expanded=not brief):
            st.code(prompt, language="markdown")
