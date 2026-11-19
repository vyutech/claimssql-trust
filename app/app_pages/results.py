"""Results page: the benchmark's headline numbers and how the system works."""

import streamlit as st

from ui_common import figure_path, load_json

st.title("Can a small model know when not to answer?", icon=":material/fact_check:")

summary = load_json("results.json")
models = summary.get("models", [])
if not models:
    st.info("No results yet. Build them with `python scripts/build_demo_db.py`.",
            icon=":material/info:")
    st.stop()

c0_counts = [m["harmful_c0"] for m in models]
c4_counts = [m["harmful_c4"] for m in models]
st.markdown(
    f"{len(models)} open models answered **200 hand-written questions** about synthetic "
    "Medicare claims, 80 of which cannot be answered from the data. On the "
    f"{summary['n_test']} test questions, **{min(c0_counts):.0f}–{max(c0_counts):.0f}** got a "
    "harmful answer without checks (**C0**), and "
    f"**{min(c4_counts):.0f}–{max(c4_counts):.0f}** with all three checks (**C4**)."
)

st.subheader("Harmful answers on the test set", icon=":material/shield:")
st.caption(f"{summary['n_test']} test questions. A harmful answer is a wrong answer, or any "
           "answer to a question that cannot be answered. Lower is better.")
cards = st.columns(len(models))
for column, model in zip(cards, models, strict=True):
    c0, c4 = model["harmful_c0"], model["harmful_c4"]
    change = (c4 - c0) / c0 * 100 if c0 else 0.0
    with column.container(border=True, height="stretch"):
        st.markdown(f"**{model['model']}**")
        st.metric("Harmful answers, C0 → C4", f"{c0:.0f} → {c4:.0f}", f"{change:.0f}%",
                  delta_color="inverse",
                  help=f"95% interval for C4: {model['harmful_c4_lo']:.0f} to "
                       f"{model['harmful_c4_hi']:.0f}")  # fmt: skip
        st.metric("Reliability Score RS₁₀, C0 → C4",
                  f"{model['rs10_c0']:.0f} → {model['rs10_c4']:.0f}",
                  f"{model['rs10_c4'] - model['rs10_c0']:+.0f}",
                  help="+1 for a right answer or a justified refusal, 0 for an unneeded "
                       "refusal, -10 for a harmful answer; times 100, averaged.")  # fmt: skip
        st.caption(f"C4 answers {model['coverage_c4']:.0%} of questions "
                   f"(agreement threshold τ = {model['tau']:g}).")

st.subheader("How it works", icon=":material/account_tree:")
st.mermaid_chart(
    """
    flowchart LR
        Q[Question] --> G[Model writes SQL]
        G --> V1{V1<br/>names exist?}
        V1 -->|yes| V2{V2<br/>SQL runs?}
        V2 -->|yes| V3{V3<br/>samples agree?}
        V3 -->|yes| A([Answer])
        G -.->|model refuses| R([Refuse with a reason])
        V1 -.->|no| R
        V2 -.->|no| R
        V3 -.->|no| R
    """,
    alt="Flowchart: the model writes SQL; it must pass V1 (all tables and columns exist), "
        "V2 (the SQL runs) and V3 (extra samples agree) to be answered, otherwise the "
        "system refuses with a reason.",
)  # fmt: skip
st.caption("C0 answers with whatever the model writes. C4 runs every check: any failed check "
           "turns the answer into a refusal. All results come from one logged run; the model "
           "is never asked again.")

st.subheader("Harmful answers by configuration", icon=":material/bar_chart:")
path = figure_path("fig7_2_harmful_by_config.png")
if path:
    left, _ = st.columns([3, 1])
    with left.container(border=True):
        st.image(str(path), width="stretch",
                 alt="Bar chart: harmful answers for configurations C0 to C4 for each model, "
                     "falling from about 75 at C0 to between 4 and 23 at C4, with 95% "
                     "confidence intervals.")  # fmt: skip
    st.caption("C1 lets the model refuse; C2 adds the schema check (V1); C3 the dry run (V2); "
               "C4 the agreement check (V3). Error bars: 95% bootstrap intervals.")
